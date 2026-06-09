"""FROZEN CONTRACT §4.5 — write_event_wav (GUANO).  STUB, frozen at M5.

Required GUANO fields: Timestamp, Samplerate, Make/Model (=UltraMic 250K), Length,
(optional) Loc Position (GPS), Temperature. Filenames include date/time
(BTO Acoustic Pipeline compatible naming).
"""

from __future__ import annotations

import numpy as np


def write_event_wav(samples: np.ndarray, fs: int, meta: dict, path: str) -> None:
    """High-rate WAV + embedded GUANO metadata (guano-py).  Implemented at M5."""
    raise NotImplementedError("write_event_wav is finalized & frozen at M5")
