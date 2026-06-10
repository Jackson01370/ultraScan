"""Incremental STFT for the live display (L2, DESIGN §3).

This is the *display* path: nfft / hop / window are free display-side choices
and deliberately decoupled from the frozen ``spec_audio.render`` contract
(DESIGN §4.1). Columns are produced continuously across ``push`` calls by
carrying the window-overlap tail between calls, so framing is identical
whether samples arrive in one block or many (continuity is testable).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class StftStream:
    """Streaming one-sided STFT: real samples in, dB magnitude columns out.

    ``push`` returns an ``(n_new_columns, nfft//2 + 1)`` float32 array of
    dB-full-scale magnitudes (0 dBFS == full-scale sine), or an empty array if
    not enough samples have accumulated for a full window yet. Call ``reset``
    whenever the upstream reader reports dropped samples — the overlap carry is
    no longer contiguous and must be rebuilt.
    """

    def __init__(self, rate: float, nfft: int = 2048, hop: Optional[int] = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if nfft < 16:
            raise ValueError("nfft too small")
        self.rate = float(rate)
        self.nfft = int(nfft)
        self.hop = int(hop) if hop else self.nfft // 2
        if not 0 < self.hop <= self.nfft:
            raise ValueError("hop must be in (0, nfft]")
        self._window = np.hanning(self.nfft).astype(np.float64)
        # full-scale sine -> coherent gain of sum(w)/2 -> 0 dBFS reference
        self._ref = float(self._window.sum()) / 2.0
        self._carry = np.empty(0, dtype=np.float32)

    @property
    def n_bins(self) -> int:
        return self.nfft // 2 + 1

    @property
    def freqs_hz(self) -> np.ndarray:
        """Center frequency of each output bin (0 .. rate/2)."""
        return np.fft.rfftfreq(self.nfft, 1.0 / self.rate)

    @property
    def columns_per_second(self) -> float:
        return self.rate / self.hop

    def reset(self) -> None:
        """Drop the overlap carry (call after the ring reader reports a gap)."""
        self._carry = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Consume new contiguous samples, return finished dB columns."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        buf = np.concatenate([self._carry, samples]) if self._carry.size else samples
        if buf.size < self.nfft:
            self._carry = buf.copy()
            return np.empty((0, self.n_bins), dtype=np.float32)

        n_cols = 1 + (buf.size - self.nfft) // self.hop
        starts = np.arange(n_cols, dtype=np.int64) * self.hop
        frames = buf[starts[:, None] + np.arange(self.nfft)] * self._window
        mag = np.abs(np.fft.rfft(frames, axis=1)) / self._ref
        db = 20.0 * np.log10(mag + 1e-10)

        consumed = n_cols * self.hop
        self._carry = buf[consumed:].copy()  # tail still inside the next window
        return db.astype(np.float32)
