# SPDX-License-Identifier: Apache-2.0

"""Lookup metrics subscriber — OTel counters for external-KV token hit rate.

Exposes counters driven by the ``MP_LOOKUP_PREFETCH_END`` event.  The
combined pair ``lookup_requested`` / ``lookup_hit`` gives the fraction of
tokens requested by a lookup that were served from LMCache. vLLM's own
prefix cache is outside this metric; optional persistent LMCache L0 is included.

    rate(lmcache_mp_lookup_hit_tokens_total[5m])
    / rate(lmcache_mp_lookup_requested_tokens_total[5m])

For legacy L1/L2 lookups, ``lookup_hit_l1`` / ``lookup_hit_l2`` split the
hit exactly. Non-monotonic L0-union lookups increment ``lookup_hit_l0_union``
instead of fabricating a tier split. ``lookups`` counts completed lookups and
``lookup_early_exit`` those that short-circuited before a cache probe,
labeled by ``reason``.

All counters carry ``model_name`` and ``cache_salt`` attributes so they can
be sliced per model and per tenant / isolation domain on the dashboard.

See ``docs/design/v1/mp_observability/L1_L2_HIT_RATE_PLAN.md`` for the full
rationale behind co-locating numerator and denominator on a single event.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
from opentelemetry import metrics

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber


def _lookup_attrs(event: Event) -> dict[str, Any]:
    """Build ``{"model_name": ..., "cache_salt": ...}`` from the event.

    Missing fields are dropped from the returned dict so future emission
    sites that haven't been updated to populate them won't crash; the
    counter just records without that label dimension.
    """
    attrs: dict[str, Any] = {}
    model_name = event.metadata.get("model_name")
    if model_name is not None:
        attrs["model_name"] = str(model_name)
    cache_salt = event.metadata.get("cache_salt")
    if cache_salt is not None:
        attrs["cache_salt"] = str(cache_salt)
    return attrs


class LookupMetricsSubscriber(EventSubscriber):
    """Maintains OTel counters for LMCache token-level cache hit rate.

    Metrics (all labeled by ``model_name`` and ``cache_salt``):
    - ``lmcache_mp.lookup_requested`` — tokens submitted for lookup
      (denominator).  Counts only the chunk-aligned portion; sub-chunk
      trailing tokens are excluded because they cannot hit by design.
    - ``lmcache_mp.lookup_hit`` — tokens found in LMCache during the
      lookup (numerator). Counts the contiguous prefix hit only.
    - ``lmcache_mp.lookup_hit_l1`` — of lookup_hit, tokens L1 could serve
      on its own under each object group's attention-window rule.
    - ``lmcache_mp.lookup_hit_l2`` — for legacy L1/L2 lookup, tokens L2
      added beyond the L1-servable prefix. There, ``l1 + l2 == lookup_hit``.
    - ``lmcache_mp.lookup_hit_l0_union`` — total hit tokens for an L0/L1/L2
      union lookup; deliberately not a fabricated per-tier split.
    - ``lmcache_mp.lookups`` — completed lookups (denominator for
      ``lookup_early_exit``).
    - ``lmcache_mp.lookup_early_exit`` — lookups that exited before a
      cache probe; extra ``reason`` label (``no_gpu_context``,
      ``empty_chunk_hashes``, ``no_group_layout_descs``).
    """

    def __init__(self) -> None:
        meter = metrics.get_meter("lmcache.lookup")

        self._requested_tokens = meter.create_counter(
            "lmcache_mp.lookup_requested",
            description=(
                "Total tokens submitted for lookup (denominator of the "
                "LMCache token-level hit rate). Only chunk-aligned tokens "
                "are counted."
            ),
            unit="tokens",
        )
        self._hit_tokens = meter.create_counter(
            "lmcache_mp.lookup_hit",
            description=(
                "Total tokens found in LMCache during lookup (numerator of "
                "the LMCache token-level hit rate). Counts the contiguous "
                "prefix hit only."
            ),
            unit="tokens",
        )
        self._l1_hit_tokens = meter.create_counter(
            "lmcache_mp.lookup_hit_l1",
            description=(
                "Of lookup_hit: tokens L1 could serve on its own under each "
                "object group's attention-window rule (from "
                "PrefetchHandle.l1_hit_chunks)."
            ),
            unit="tokens",
        )
        self._l2_hit_tokens = meter.create_counter(
            "lmcache_mp.lookup_hit_l2",
            description=(
                "Of lookup_hit: tokens L2 added beyond the L1-servable "
                "prefix. lookup_hit_l1 + lookup_hit_l2 == lookup_hit."
            ),
            unit="tokens",
        )
        self._l0_union_hit_tokens = meter.create_counter(
            "lmcache_mp.lookup_hit_l0_union",
            description=(
                "Total contiguous hit tokens for lookups using the "
                "non-monotonic persistent-L0/L1/L2 union path. This is not "
                "a per-tier attribution."
            ),
            unit="tokens",
        )
        self._lookups = meter.create_counter(
            "lmcache_mp.lookups",
            description="Completed lookups (denominator for lookup_early_exit).",
            unit="requests",
        )
        self._early_exits = meter.create_counter(
            "lmcache_mp.lookup_early_exit",
            description=(
                "Lookups that exited before a normal cache probe, labeled "
                "by reason (no_gpu_context, empty_chunk_hashes, "
                "no_group_layout_descs)."
            ),
            unit="requests",
        )

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        return {
            EventType.MP_LOOKUP_PREFETCH_END: self._on_lookup_prefetch_end,
        }

    def _on_lookup_prefetch_end(self, event: Event) -> None:
        attrs = _lookup_attrs(event)
        self._requested_tokens.add(event.metadata["requested_tokens"], attributes=attrs)
        self._hit_tokens.add(event.metadata["hit_tokens"], attributes=attrs)
        self._lookups.add(1, attributes=attrs)
        # .get(): tolerate events from emitters predating the attribution
        # fields; they simply do not move the split counters.
        if event.metadata.get("l0_union_enabled", False):
            self._l0_union_hit_tokens.add(
                event.metadata.get("hit_tokens", 0), attributes=attrs
            )
        else:
            self._l1_hit_tokens.add(
                event.metadata.get("l1_hit_tokens", 0), attributes=attrs
            )
            self._l2_hit_tokens.add(
                event.metadata.get("l2_hit_tokens", 0), attributes=attrs
            )
        early_exit_reason = event.metadata.get("early_exit_reason", "")
        if early_exit_reason:
            self._early_exits.add(1, attributes={**attrs, "reason": early_exit_reason})
