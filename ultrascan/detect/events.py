"""FROZEN CONTRACT §4.4 — Event dataclass (FROZEN at M4).

The field set / types ARE the frozen §4.4 contract — changing them needs human
approval (DESIGN §4, §11). The M4 measurement code that fills these fields is
``AdaptiveSnrDetector`` (see detector.py).

Notes on the fields:
  * ``duration`` is deliberately NOT a field — it is ``t_end - t_start`` (s).
  * ``slope`` is kHz/ms (peak-frequency sweep rate; ~0 for a steady tone).
  * ``n_pulses`` / ``ipi``: the M4 detector emits one time-segment per Event
    (``n_pulses=1``, ``ipi=None``); pulse-train decomposition (n_pulses>1, IPI)
    is left to a future detector — the fields are reserved here so the frozen
    shape already carries them.
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
