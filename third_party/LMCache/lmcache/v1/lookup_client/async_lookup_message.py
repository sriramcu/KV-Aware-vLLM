# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Optional

# Third Party
import msgspec


class AsyncLookupMsg(msgspec.Struct, tag=True):  # type: ignore
    """Base class for async lookup messages"""

    def describe(self) -> str:
        return ""


class LookupRequestMsg(AsyncLookupMsg):
    """Async lookup request message from scheduler to worker"""

    lookup_id: str
    hashes: list[int]
    offsets: list[int]
    request_configs: Optional[Dict[str, str]] = None

    def describe(self) -> str:
        return (
            f"Async lookup request for lookup_id={self.lookup_id} "
            f"with {len(self.hashes)} hashes"
        )


class LookupResponseMsg(AsyncLookupMsg):
    """Async lookup response message from worker to scheduler"""

    lookup_id: str
    num_hit_tokens: int

    def describe(self) -> str:
        return (
            f"Async lookup response for lookup_id={self.lookup_id} "
            f"with {self.num_hit_tokens} hit tokens"
        )


class LookupCleanupMsg(AsyncLookupMsg):
    """Cleanup message from scheduler to worker to release memory objects"""

    lookup_id: str

    def describe(self) -> str:
        return f"Cleanup memory for lookup_id={self.lookup_id}"

class LookupPVTSCMsg(AsyncLookupMsg):
    """
    Post-vLLM-timeout synthetic-completion request.

    The scheduler sends this only after its poll-based async lookup timeout has
    returned zero LMCache tokens to vLLM. Workers should stop future disk work
    for this lookup at safe boundaries and complete the disk tier as an empty
    result. This is intentionally distinct from LookupCleanupMsg, which runs
    after prefetch completion and releases returned memory objects.
    """

    lookup_id: str

    def describe(self) -> str:
        return f"PVTSC soft-stop for lookup_id={self.lookup_id}"

