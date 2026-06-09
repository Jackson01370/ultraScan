"""FROZEN CONTRACT §4.2 — Audifier (swappable audification component).

STUB (M0). The Protocol is fixed here; the first implementation
(`HeterodyneAudifier`, image-free DDC chain per §5) lands and is frozen at M2.

Implementation order: HeterodyneAudifier (v1) -> TimeExpansionAudifier (v2)
                      -> PhaseVocoderAudifier (optional).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Audifier(Protocol):
    def configure(self, f_lo_sel: float, bandwidth: float, fs_in: float) -> None:
        """Set selected band lower edge, bandwidth, and input rate."""
        ...

    def process(self, block: np.ndarray) -> np.ndarray:
        """Raw sample block (real, fs_in) -> audible block (real, fs_out).

        State (NCO phase, FIR `zi`) MUST persist across blocks — a discontinuity
        at a block boundary produces an audible click (DESIGN §5).
        """
        ...
