# SPDX-License-Identifier: Apache-2.0
"""[SC] Optional native CUDA phase-level H2D diagnostic logger.

Purely observational: never feeds the Bayesian placement estimator. Native
samples use CUDA events and distinguish STAGING (H2D DMA) from KERNEL
(scatter into paged KV). See EventType.MP_TRANSFER_PHASE_SAMPLES.
"""

from __future__ import annotations

from lmcache.logging import init_logger
from lmcache.lmcache_native import TransferDirection
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventSubscriber
from lmcache.v1.platform.ops_types import TransferPhase

logger = init_logger(__name__)


class BayesH2DDiagnosticSubscriber(EventSubscriber):
    def __init__(self) -> None:
        # Verify the installed native extension supports stage-level timing.
        from lmcache import device_ops
        self.native_available = hasattr(device_ops, "pop_completed_phase_timings")
        logger.info(
            "[BAYES_H2D_DIAG_INIT] native_phase_timing=%d",
            int(self.native_available),
        )

    def get_subscriptions(self):
        return {EventType.MP_TRANSFER_PHASE_SAMPLES: self._on_samples}

    def _on_samples(self, event: Event) -> None:
        if not self.native_available:
            return
        for sample in event.metadata.get("samples", ()):
            if len(sample) != 8:
                continue
            phase, direction, gpu_idx, elapsed_ms, nbytes, transfer_key, start_s, end_s = sample
            if int(direction) != int(TransferDirection.H2D):
                continue
            phase_name = {
                int(TransferPhase.STAGING): "staging_dma",
                int(TransferPhase.KERNEL): "scatter_kernel",
            }.get(int(phase), f"unknown_{phase}")
            logger.info(
                "[BAYES_H2D_PHASE] transfer_key=%s device=cuda:%s "
                "phase=%s elapsed_ms=%.6f nbytes=%d start_s=%.6f end_s=%.6f",
                transfer_key, gpu_idx, phase_name, elapsed_ms, nbytes,
                start_s, end_s,
            )
