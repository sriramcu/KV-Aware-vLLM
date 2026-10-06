# SPDX-License-Identifier: Apache-2.0
# [SC] Project-specific changes in this upstream file are marked with [SC];
# see repo-root docs/SC_MODIFICATIONS.md for rationale and provenance.

"""
Store policy interface and default implementation for L1-to-L2 storage decisions.

The store policy makes two decisions after data is written to L1:
1. Which L2 adapter(s) should each key be stored to?
2. After a successful L2 store, should the key be deleted from L1?
"""

# Standard
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

# First Party
from lmcache.logging import init_logger

from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class AdapterDescriptor:
    """
    Lightweight descriptor for an L2 adapter, giving the store policy
    enough information to distinguish adapters without exposing runtime
    objects.
    """

    index: int
    """Position in the L2 adapters list."""

    config: L2AdapterConfigBase
    """The adapter's configuration object."""

    @property
    def type_name(self) -> str:
        """
        Registered adapter type name (e.g., "mock", "disk", "redis").

        Derived from the config's registered type via reverse lookup.

        Returns:
            str: The registered type name.
        """
        return get_type_name_for_config(self.config)


class StorePolicy(ABC):
    """
    Abstract interface for store decisions.

    The store policy is called by the StoreController to decide:
    1. Which adapter(s) to store each key to (select_store_targets).
    2. Which keys to delete from L1 after successful L2 store
       (select_l1_deletions).
    """

    @abstractmethod
    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Decide which keys to store to which L2 adapters.

        Args:
            keys: Keys that were just written to L1 and are
                candidates for L2 storage.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from adapter index to list of keys to store
            to that adapter. Keys absent from all lists are NOT
            stored to L2.
        """

    @abstractmethod
    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Decide which keys to delete from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Keys to delete from L1. Empty list means keep all.
        """

    # [SC] Preserve policy ownership semantics when StoreController's rolling
    # L2 byte budget deliberately skips a store.  Default behavior mirrors a
    # successful store; cache-like policies override this by returning [].
    def select_l1_deletions_on_store_skip(
        self, keys: list[ObjectKey]
    ) -> list[ObjectKey]:
        return self.select_l1_deletions(keys)


# -----------------------------------------------------------------------------
# Registry: store policy name -> policy class
# -----------------------------------------------------------------------------

_STORE_POLICY_REGISTRY: dict[str, type[StorePolicy]] = {}


def register_store_policy(
    name: str,
    policy_cls: type[StorePolicy],
) -> None:
    """
    Register a store policy class under a name.

    Each policy module should call this at import time.

    Args:
        name: Policy name (e.g. "default").
        policy_cls: A concrete StorePolicy subclass.
    """
    if name in _STORE_POLICY_REGISTRY:
        raise ValueError(f"Store policy already registered: {name!r}")
    _STORE_POLICY_REGISTRY[name] = policy_cls


def get_registered_store_policies() -> list[str]:
    """Return the list of registered store policy names."""
    return list(_STORE_POLICY_REGISTRY)


def create_store_policy(name: str) -> StorePolicy:
    """
    Create a store policy instance by name.

    Args:
        name: Registered policy name.

    Returns:
        A new StorePolicy instance.

    Raises:
        ValueError: If no policy is registered under the given name.
    """
    if name not in _STORE_POLICY_REGISTRY:
        known = ", ".join(sorted(_STORE_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown store policy {name!r}. Known: {known}")
    return _STORE_POLICY_REGISTRY[name]()


class DefaultStorePolicy(StorePolicy):
    """
    Default store policy: store all keys to all adapters,
    never delete from L1.
    """

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        """
        Store all keys to all adapters.

        Args:
            keys: Keys that were just written to L1.
            adapters: Descriptors of available L2 adapters.

        Returns:
            Mapping from every adapter index to the full list of keys.
        """
        return {ad.index: list(keys) for ad in adapters}

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Never delete from L1.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            Empty list (keep all keys in L1).
        """
        return []


class BufferOnlyStorePolicy(DefaultStorePolicy):
    """
    Buffer-only store policy: store all keys to all adapters,
    then delete them from L1 immediately.

    Use this with NoOpEvictionPolicy to avoid useless LRU
    tracking overhead when L1 is a pure write buffer.

    Inherits ``select_store_targets`` from ``DefaultStorePolicy``
    (store all keys to all adapters) and only overrides the L1
    deletion decision.
    """

    def select_l1_deletions(
        self,
        keys: list[ObjectKey],
    ) -> list[ObjectKey]:
        """
        Delete all keys from L1 after successful L2 store.

        Args:
            keys: Keys that were successfully stored to L2.

        Returns:
            All keys (remove everything from L1).
        """
        return list(keys)


register_store_policy("default", DefaultStorePolicy)
register_store_policy("skip_l1", BufferOnlyStorePolicy)

# [SC] Short-Q dynamic placement policy. Semantic labels are gpu/cpu/disk/drop;
# LMCache itself remains a physical CPU-L1 / disk-L2 hierarchy. True drop
# objects are never admitted to LMCache persistence.
class GNNDynamicStorePolicy(StorePolicy):
    """Persist Short-Q-selected disk objects, optionally safety-back all objects.

    With ``LMCACHE_GNN_L2_BACKING=0`` only ``disk`` placements are sent to L2.
    With backing enabled, all host-resident objects get an L2 safety copy while
    only true ``disk`` placements are deleted from L1 after commit. ``drop``
    placements are not host-resident and therefore never reach this policy.
    This keeps ``gpu``/``cpu`` placement objects resident in fast host memory
    when present.
    """

    def __init__(self) -> None:
        value = os.getenv("LMCACHE_GNN_L2_BACKING", "0").strip().lower()
        self._l2_backing = value in {"1", "true", "yes", "on"}
        logger.info("[GNN_DYNAMIC_L2_BACKING_INIT] enabled=%s", self._l2_backing)

    @staticmethod
    def _placements(keys: list[ObjectKey]) -> dict[ObjectKey, str]:
        from lmcache.v1.distributed.placement_metadata import get_chunk_placement

        return {key: get_chunk_placement(key.chunk_hash) for key in keys}

    def select_store_targets(
        self,
        keys: list[ObjectKey],
        adapters: list[AdapterDescriptor],
    ) -> dict[int, list[ObjectKey]]:
        placements = self._placements(keys)
        if self._l2_backing:
            l2_keys = list(keys)
        else:
            l2_keys = [key for key in keys if placements[key] == "disk"]

        persistent_l1 = sum(
            1 for key in keys if placements[key] in {"gpu", "cpu"}
        )
        backing_targets = sum(
            1 for key in l2_keys if placements[key] in {"gpu", "cpu"}
        )
        logger.info(
            "[GNN_DYNAMIC_L2_POLICY] candidates=%d persistent_l1=%d "
            "disk_targets=%d backing_targets=%d l2_backing=%s adapters=%d",
            len(keys),
            persistent_l1,
            sum(1 for key in l2_keys if placements[key] == "disk"),
            backing_targets,
            self._l2_backing,
            len(adapters),
        )
        return {ad.index: list(l2_keys) for ad in adapters if l2_keys}

    def select_l1_deletions(self, keys: list[ObjectKey]) -> list[ObjectKey]:
        placements = self._placements(keys)
        deletions = [key for key in keys if placements[key] == "disk"]
        if keys:
            logger.info(
                "[GNN_DYNAMIC_L2_COMMIT] l2_keys=%d delete_l1_staging=%d "
                "keep_l1_backing=%d",
                len(keys),
                len(deletions),
                len(keys) - len(deletions),
            )
        return deletions

    def select_l1_deletions_on_store_skip(
        self, keys: list[ObjectKey]
    ) -> list[ObjectKey]:
        # A skipped true-disk object was admitted to L1 only as L2 staging.
        # gpu/cpu placements retain their fast-tier backing copy.
        placements = self._placements(keys)
        deletions = [key for key in keys if placements[key] == "disk"]
        if keys:
            logger.info(
                "[GNN_DYNAMIC_L2_SKIP] skipped_l2_keys=%d delete_l1_staging=%d "
                "keep_l1_backing=%d",
                len(keys),
                len(deletions),
                len(keys) - len(deletions),
            )
        return deletions


register_store_policy("gnn_dynamic", GNNDynamicStorePolicy)
