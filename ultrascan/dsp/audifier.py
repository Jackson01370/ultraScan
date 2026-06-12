"""FROZEN CONTRACT §4.2 — Audifier (swappable audification component).

╔══════════════════════════════════════════════════════════════════════════╗
║  FROZEN at M2a.  From here on `git diff --stat` for this file must stay   ║
║  empty; signature changes need human approval (DESIGN §4, §11).           ║
╚══════════════════════════════════════════════════════════════════════════╝

Implementation order: HeterodyneAudifier (v1) -> TimeExpansionAudifier (v2)
                      -> PhaseVocoderAudifier (optional). Later audifiers are
new classes in new modules — this frozen file is not edited for them.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import soxr
from scipy.signal import firwin, lfilter

# v1 fixed output rate (§5 step 6: fs_dec -> 48 kHz fractional resample).
FS_OUT = 48_000.0


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


class HeterodyneAudifier:
    """Image-free heterodyne DDC chain (DESIGN §5) — block-streaming, stateful.

    Chain per block (input real, fs_in):
      1) complex NCO  exp(-j 2π f_lo_sel n / fs_in)   — phase accumulated across blocks
      2) mix          x = s · lo                       — band lower edge lands at DC
      3) LPF          one-sided complex FIR, passband [0, cutoff],
                      cutoff = min(bandwidth, fs_dec/2) — lfilter zi carried across blocks
      4) decimate     fs_in -> fs_dec (÷DECIM, integer) — pick phase carried across blocks
      5) realify      audio = real(y_decim)
      6) soxr         fs_dec -> FS_OUT (streaming resampler, state carried across blocks)

    Why the LPF must be COMPLEX one-sided (the 解析信号 of DESIGN §1):
    with real-coefficient taps, real() commutes with the filter, so
    real(LPF(s·e^{-jωn})) ≡ LPF(s·cos(ωn)) — mathematically identical to the
    PROHIBITED real-cos mix, and a tone at the image frequency f_lo−Δ folds onto
    +Δ. The one-sided passband [0, cutoff] keeps only positive baseband
    frequencies, so the image (at −Δ) is rejected before realify (§9 mirror test).
    Passband gain is 2 so a tone's amplitude survives the analytic-signal halving.

    Validity limits enforced by ``configure`` (the filter is fs-periodic and has
    a finite skirt, so the image-free guarantee has boundaries):
      - ``f_lo_sel + cutoff <= fs_in/2``: beyond that, the mix product of a real
        tone's negative line wraps mod fs to ``fs − f − f_lo`` INSIDE the
        passband and returns at full gain — the §11 defect through the back door.
      - ``bandwidth >= one FIR transition width`` (≈ 3.3·fs_in/N_TAPS ≈ 3.2 kHz
        at 250k): narrower requests would select nothing but filter skirt.
    Skirt caveat (finite 255-tap FIR, not brick-wall): image rejection reaches
    full depth for content more than ~½ transition (≈1.6 kHz at 250k) below the
    LO; band edges roll off over the same width.

    ``configure`` may be called again at any time (band re-selection); it fully
    resets stream state, so the boundary is not click-free — acceptable for v1.
    """

    DECIM = 5      # §5: 250k -> 50k integer decimation
    N_TAPS = 255   # linear-phase FIR; ~3 kHz transition at fs_in=250k (Hamming)

    def __init__(self):
        self._configured = False

    @property
    def fs_out(self) -> float:
        return FS_OUT

    def configure(self, f_lo_sel: float, bandwidth: float, fs_in: float) -> None:
        if fs_in <= 0:
            raise ValueError(f"fs_in must be > 0, got {fs_in!r}")
        if fs_in % self.DECIM != 0:
            raise ValueError(
                f"fs_in must be an integer multiple of DECIM={self.DECIM}, got {fs_in!r}"
            )
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be > 0, got {bandwidth!r}")
        if not 0.0 <= f_lo_sel < fs_in / 2.0:
            raise ValueError(f"f_lo_sel must be in [0, fs_in/2), got {f_lo_sel!r}")
        min_bw = 3.3 * fs_in / self.N_TAPS  # Hamming FIR transition width
        if bandwidth < min_bw:
            raise ValueError(
                f"bandwidth {bandwidth!r} is below the FIR transition width "
                f"({min_bw:.0f} Hz at fs_in={fs_in:.0f}): nothing but skirt would pass"
            )

        fs_dec = float(fs_in) / self.DECIM
        cutoff = min(float(bandwidth), fs_dec / 2.0)
        if f_lo_sel + cutoff > fs_in / 2.0:
            raise ValueError(
                f"selection [f_lo, f_lo+cutoff] = [{f_lo_sel:.0f}, "
                f"{f_lo_sel + cutoff:.0f}] Hz exceeds Nyquist "
                f"{fs_in / 2.0:.0f} Hz: the wrapped image fs−f−f_lo would land "
                f"inside the passband at full gain (see class docstring)"
            )

        # All validation passed — only now mutate, so a refused re-configure
        # leaves a running stream fully intact (atomic configure).
        self.f_lo_sel = float(f_lo_sel)
        self.bandwidth = float(bandwidth)
        self.fs_in = float(fs_in)
        self.fs_dec = fs_dec
        self.cutoff = cutoff

        # One-sided complex FIR: real lowpass at cutoff/2, modulated up by
        # cutoff/2 -> passband [0, cutoff]. Gain 2 (see class docstring).
        half = self.cutoff / 2.0
        b = firwin(self.N_TAPS, half, fs=self.fs_in)
        k = np.arange(self.N_TAPS)
        self._taps = 2.0 * b * np.exp(2j * np.pi * half * k / self.fs_in)

        # Stream state — everything that must survive block boundaries (§5).
        self._zi = np.zeros(self.N_TAPS - 1, dtype=np.complex128)
        self._phase = 0.0
        self._dphi = 2.0 * np.pi * self.f_lo_sel / self.fs_in
        self._decim_offset = 0
        self._resampler = soxr.ResampleStream(self.fs_dec, FS_OUT, 1, dtype="float32")
        self._configured = True

    def process(self, block: np.ndarray) -> np.ndarray:
        if not self._configured:
            raise RuntimeError("configure() must be called before process()")
        s = np.asarray(block, dtype=np.float64).reshape(-1)
        if s.size == 0:
            return np.empty(0, dtype=np.float32)
        n = s.size

        # 1-2) NCO + mix (phase continues exactly where the last block ended)
        ph = self._phase + self._dphi * np.arange(n)
        x = s * np.exp(-1j * ph)
        self._phase = float((self._phase + self._dphi * n) % (2.0 * np.pi))

        # 3) one-sided LPF with carried state
        y, self._zi = lfilter(self._taps, 1.0, x, zi=self._zi)

        # 4) integer decimation with carried pick phase
        picks = y[self._decim_offset::self.DECIM]
        self._decim_offset = (self._decim_offset - n) % self.DECIM

        # 5) realify  6) fractional resample to FS_OUT (streaming, no flush)
        audio = np.real(picks).astype(np.float32)
        return self._resampler.resample_chunk(audio)

    def flush(self) -> np.ndarray:
        """Drain the resampler tail and END the stream (offline/one-shot use).

        Live streaming never calls this. After flush() the instance needs a new
        configure() before further process() calls. The FIR group delay
        (~(N_TAPS-1)/2 input samples, ≈0.5 ms) is not zero-padded out — only the
        samples already inside the soxr stream are recovered.
        """
        if not self._configured:
            raise RuntimeError("configure() must be called before flush()")
        tail = self._resampler.resample_chunk(np.empty(0, dtype=np.float32), last=True)
        self._configured = False  # soxr stream is finished; force reconfigure
        return tail
