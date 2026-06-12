"""DDC chain — image-free heterodyne (DESIGN §5).

The streaming implementation (NCO phase / FIR zi / decimation phase / soxr state
all carried across blocks) lives in :class:`ultrascan.dsp.audifier.HeterodyneAudifier`
(FROZEN CONTRACT §4.2). This module keeps an offline one-shot convenience used by
tests and analysis.

PROHIBITED (§11): real cos-only mix — images fold back. Must be complex mix + LPF.
"""

from __future__ import annotations

import numpy as np

from ultrascan.dsp.audifier import HeterodyneAudifier


def ddc_heterodyne(
    samples: np.ndarray, fs_in: float, f_lo_sel: float, bandwidth: float
) -> np.ndarray:
    """Offline one-shot DDC: whole buffer in, audible-rate audio out (real, 48 kHz).

    Flushes the streaming resampler so the buffer tail is not dropped (review
    finding) — that is the one-shot/offline difference from live streaming.
    """
    aud = HeterodyneAudifier()
    aud.configure(f_lo_sel, bandwidth, fs_in)
    body = aud.process(samples)
    return np.concatenate([body, aud.flush()])
