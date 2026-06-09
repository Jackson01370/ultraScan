"""FROZEN CONTRACT §4.4 — Event dataclass.

STUB context (M0): the field layout is fixed here; the measurement code that fills
these fields lands at M4. Field set / types are part of the frozen contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Event:
    t_start: float
    t_end: float
    f_peak: float
    f_min: float
    f_max: float
    bandwidth: float
    slope: float            # kHz/ms
    n_pulses: int
    ipi: Optional[float]    # inter-pulse interval (s); None if single pulse
    snr_db: float
