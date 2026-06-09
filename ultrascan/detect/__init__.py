"""L5 event detection / measurement.

M0: signature stubs only. `Detector` is finalized & frozen at M4.
The `Event` dataclass shape (§4.4) is defined now so downstream code can import it.
"""

from .events import Event

__all__ = ["Event"]
