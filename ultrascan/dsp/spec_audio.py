"""FROZEN CONTRACT §4.1 — spectrogram render (the bridge to the CNN).

╔══════════════════════════════════════════════════════════════════════════╗
║  FROZEN at M1.  From here on `git diff --stat` for this file must stay    ║
║  empty; signature OR behaviour changes need human approval (DESIGN §4,    ║
║  §11).  The CNN input must always come from this function — no extra      ║
║  preprocessing, resize, or normalization downstream (sigscan discipline). ║
╚══════════════════════════════════════════════════════════════════════════╝

This is the real-input sister of sigscan's ``spec.render`` (which takes I/Q).
Output shape is kept identical ([256,256] float32 in [0,1]) so sigscan's CNN /
training harness can be reused unchanged in v2.

Canonical computation (fixed by this freeze):
  - window:   Hann (``np.hanning``), NFFT = 512, one-sided ``rfft``.
  - bins:     rows 0..255 = rfft bins 0..255 (DC .. 255/512 * rate; the
              Nyquist-most bin 256 is dropped to land exactly on 256 rows).
  - frames:   256 window positions evenly spaced over the whole input
              (``linspace(0, N-512, 256)``, rounded) — any input length maps
              deterministically onto 256 columns with no resize step.
              Inputs shorter than 512 samples are zero-padded to one frame.
  - scale:    magnitude in dB relative to the loudest bin (0 dB), clipped at
              -DB_DYN_RANGE, mapped linearly onto [0, 1].
  - layout:   out[i, j] = freq bin i (ascending, row 0 = DC) at time column j
              (ascending, column 0 = oldest). All-zero input -> all-zero output.
"""

from __future__ import annotations

import numpy as np

# Output contract — must match sigscan. Do not "improve" these in v1 (DESIGN §4.1).
OUT_SHAPE = (256, 256)
DB_DYN_RANGE = 60.0
NFFT = 512  # one-sided rfft -> 257 bins; rows keep bins 0..255


def render(samples: np.ndarray, rate: float) -> np.ndarray:
    """Real samples -> [256,256] float32 [0,1] spectrogram (rfft, one-sided).

    Args:
        samples: 1-D real-valued samples (any length >= 1).
        rate: sample rate in Hz (> 0). Fixes the physical frequency axis:
              row i spans frequency i * rate / NFFT.

    Returns:
        ``np.ndarray`` of shape (256, 256), dtype float32, values in [0, 1].
        Rows = frequency (row 0 = DC), columns = time (column 0 = oldest).

    NOTE: CNN input must always come from this function — no extra preprocessing,
    resize, or normalization is allowed downstream (sigscan discipline, DESIGN §4.1).
    """
    if rate <= 0:
        raise ValueError(f"rate must be > 0, got {rate!r}")
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size == 0:
        raise ValueError("samples must not be empty")

    n_freq, n_time = OUT_SHAPE
    if x.size < NFFT:
        x = np.pad(x, (0, NFFT - x.size))

    starts = np.round(np.linspace(0.0, x.size - NFFT, n_time)).astype(np.int64)
    window = np.hanning(NFFT)
    frames = x[starts[:, None] + np.arange(NFFT)] * window      # (time, NFFT)
    mag = np.abs(np.fft.rfft(frames, axis=1))[:, :n_freq]       # (time, freq)

    ref = float(mag.max())
    if ref <= 0.0:
        return np.zeros(OUT_SHAPE, dtype=np.float32)
    floor = ref * 10.0 ** (-DB_DYN_RANGE / 20.0)
    db = 20.0 * np.log10(np.maximum(mag, floor) / ref)          # [-DB_DYN_RANGE, 0]
    out = (db + DB_DYN_RANGE) / DB_DYN_RANGE                    # [0, 1]
    return np.ascontiguousarray(out.T, dtype=np.float32)        # (freq, time)
