"""FROZEN CONTRACT §4.4 — Detector Protocol.  STUB, finalized & frozen at M4.

Rule (§4.4, §11): a rule/heuristic detector's output must NOT be used as CNN
training labels (label-noise prevention; sigscan discipline).
"""

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from .events import Event


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame_or_buffer) -> List[Event]:
        """SNR / energy based event detection + segmentation."""
        ...
