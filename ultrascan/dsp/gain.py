"""FROZEN CONTRACT §4.3 — GainStage (small-signal amplification).

STUB (M0). Protocol fixed here; implementations (`AGCGain` [continuous default],
`NormalizeGain` [snapshot], `CompressorGain`) land and are frozen at M3.

Reminder (DESIGN §4.3): display gain (waterfall contrast) and audio gain (volume)
are SEPARATE knobs — never conflate them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GainStage(Protocol):
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Audio block in -> level-adjusted audio block out."""
        ...
