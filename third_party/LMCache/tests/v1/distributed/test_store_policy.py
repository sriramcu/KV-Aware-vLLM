# SPDX-License-Identifier: Apache-2.0
# [SC] Project-specific changes in this upstream file are marked with [SC];
# see repo-root docs/SC_MODIFICATIONS.md for rationale and provenance.

"""
Unit tests for store policy interface and DefaultStorePolicy.

Tests are written against the StorePolicy contract defined in store_policy.py.
"""

# Third Party

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.store_policy import (
    AdapterDescriptor,
    DefaultStorePolicy,
)

# =============================================================================
# Helpers
# =============================================================================


def make_object_key(chunk_id: int) -> ObjectKey:
    """Create a test ObjectKey with the given chunk ID."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test_model",
        kv_rank=0,
    )


def make_descriptor(index: int) -> AdapterDescriptor:
    """Create an AdapterDescriptor for testing."""
    config = MockL2AdapterConfig(max_size_gb=1.0, mock_bandwidth_gb=10.0)
    return AdapterDescriptor(index=index, config=config)


# =============================================================================
# DefaultStorePolicy Tests
# =============================================================================


class TestDefaultStorePolicyTargets:
    """Test DefaultStorePolicy.select_store_targets behavior."""

    def test_single_adapter_all_keys(self):
        """All keys should be sent to the single adapter."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        assert 0 in result
        assert result[0] == keys

    def test_multiple_adapters_all_keys_to_each(self):
        """All keys should be sent to every adapter."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(3)]
        adapters = [make_descriptor(0), make_descriptor(1)]

        result = policy.select_store_targets(keys, adapters)

        assert len(result) == 2
        assert result[0] == keys
        assert result[1] == keys

    def test_empty_adapters_returns_empty(self):
        """No adapters means no store targets."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(0)]

        result = policy.select_store_targets(keys, [])

        assert result == {}

    def test_empty_keys_returns_empty_lists(self):
        """Empty keys list should produce empty lists for each adapter."""
        policy = DefaultStorePolicy()
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets([], adapters)

        assert 0 in result
        assert result[0] == []

    def test_returns_copies_not_references(self):
        """Returned lists should be independent copies of the input."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(0)]
        adapters = [make_descriptor(0)]

        result = policy.select_store_targets(keys, adapters)

        # Mutating the result should not affect the input
        result[0].append(make_object_key(99))
        assert len(keys) == 1


class TestDefaultStorePolicyDeletions:
    """Test DefaultStorePolicy.select_l1_deletions behavior."""

    def test_never_deletes(self):
        """DefaultStorePolicy should never delete from L1."""
        policy = DefaultStorePolicy()
        keys = [make_object_key(i) for i in range(5)]

        result = policy.select_l1_deletions(keys)

        assert result == []

    def test_empty_keys_returns_empty(self):
        """Empty input should return empty output."""
        policy = DefaultStorePolicy()

        result = policy.select_l1_deletions([])

        assert result == []


# [SC] KV-Aware semantic placement/store-budget regression coverage.
class TestGNNDynamicStorePolicyL2Backing:
    """GNN L2 backing preserves placement while adding a durable copy."""

    def test_backing_targets_every_key_but_only_deletes_true_disk(
        self, monkeypatch
    ):
        from lmcache.v1.distributed import placement_metadata
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            GNNDynamicStorePolicy,
        )

        keys = [make_object_key(i) for i in range(3)]
        placements = {
            keys[0].chunk_hash: "gpu",
            keys[1].chunk_hash: "cpu",
            keys[2].chunk_hash: "disk",
        }
        monkeypatch.setattr(
            placement_metadata,
            "get_chunk_placement",
            lambda chunk_hash: placements[chunk_hash],
        )
        monkeypatch.setenv("LMCACHE_GNN_L2_BACKING", "1")

        policy = GNNDynamicStorePolicy()
        result = policy.select_store_targets(keys, [make_descriptor(0)])

        assert result[0] == keys
        assert policy.select_l1_deletions(keys) == [keys[2]]

    def test_default_dynamic_policy_keeps_selective_disk_behavior(
        self, monkeypatch
    ):
        from lmcache.v1.distributed import placement_metadata
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            GNNDynamicStorePolicy,
        )

        keys = [make_object_key(i) for i in range(3)]
        placements = {
            keys[0].chunk_hash: "gpu",
            keys[1].chunk_hash: "cpu",
            keys[2].chunk_hash: "disk",
        }
        monkeypatch.setattr(
            placement_metadata,
            "get_chunk_placement",
            lambda chunk_hash: placements[chunk_hash],
        )
        monkeypatch.delenv("LMCACHE_GNN_L2_BACKING", raising=False)

        policy = GNNDynamicStorePolicy()
        result = policy.select_store_targets(keys, [make_descriptor(0)])

        assert result[0] == [keys[2]]
        assert policy.select_l1_deletions([keys[2]]) == [keys[2]]
    def test_store_budget_skip_deletes_only_disk_staging(self, monkeypatch):
        from lmcache.v1.distributed import placement_metadata
        from lmcache.v1.distributed.storage_controllers.store_policy import (
            GNNDynamicStorePolicy,
        )

        keys = [make_object_key(i) for i in range(3)]
        placements = {
            keys[0].chunk_hash: "gpu",
            keys[1].chunk_hash: "cpu",
            keys[2].chunk_hash: "disk",
        }
        monkeypatch.setattr(
            placement_metadata,
            "get_chunk_placement",
            lambda chunk_hash: placements[chunk_hash],
        )

        policy = GNNDynamicStorePolicy()
        assert policy.select_l1_deletions_on_store_skip(keys) == [keys[2]]
