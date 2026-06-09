"""FROZEN CONTRACT §4.1 — spectrogram render (the bridge to the CNN).

╔══════════════════════════════════════════════════════════════════════════╗
║  STUB (M0).  Signature is fixed here; the body is implemented and the     ║
║  file is FROZEN at M1.  From M1 on, `git diff --stat` for this file must   ║
║  stay empty (signature changes need human approval — DESIGN §4, §11).      ║
╚══════════════════════════════════════════════════════════════════════════╝

This is the real-input sister of sigscan's ``spec.render`` (which takes I/Q).
Output shape is kept identical ([256,256] float32 in [0,1]) so sigscan's CNN /
training harness can be reused unchanged in v2.
"""

from __future__ import annotations

import numpy as np

# Output contract — must match sigscan. Do not "improve" these in v1 (DESIGN §4.1).
OUT_SHAPE = (256, 256)
DB_DYN_RANGE = 60.0


def render(samples: np.ndarray, rate: float) -> np.ndarray:
    """Real samples -> [256,256] float32 [0,1] spectrogram (rfft, one-sided).

    Args:
        samples: 1-D real-valued samples.
        rate: sample rate in Hz.

    Returns:
        ``np.ndarray`` of shape (256, 256), dtype float32, values in [0, 1].

    NOTE: CNN input must always come from this function — no extra preprocessing,
    resize, or normalization is allowed downstream (sigscan discipline, DESIGN §4.1).
    """
    raise NotImplementedError("spec_audio.render is finalized & frozen at M1")
