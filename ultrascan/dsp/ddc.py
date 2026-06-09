"""DDC chain — image-free heterodyne (DESIGN §5).  STUB, implemented at M2.

Chain (input real s[n], fs=250k; selected band [f_lo_sel, f_hi_sel], W = hi - lo):
  1) complex NCO:  lo[n] = exp(-j 2pi f_lo_sel n / fs)   # phase accumulates across blocks
  2) mix:          x[n]  = s[n] * lo[n]                   # band lower edge -> DC, complex baseband
  3) LPF:          y[n]  = lowpass(x, cutoff=min(W, fs_dec/2))  # FIR, keep zi across blocks
  4) decimate:     250k -> 50k (/5, integer); LPF doubles as anti-alias
  5) realify:      audio[n] = real(y_decim[n])
  6) soxr:         50k -> 48k (fractional, high quality)

PROHIBITED (§11): real cos-only mix — images fold back. Must be complex mix + LPF.
"""

from __future__ import annotations

import numpy as np


def ddc_heterodyne(*args, **kwargs) -> np.ndarray:  # pragma: no cover - stub
    raise NotImplementedError("DDC chain is implemented at M2 (DESIGN §5)")
