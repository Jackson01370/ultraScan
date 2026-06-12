"""SPSC audio ring (DESIGN §3 L3 -> L0'): one producer, one consumer, strict FIFO.

Single producer = the audification worker (pushes processed 48 kHz audio).
Single consumer = the output callback (drains; copy + counters ONLY — DESIGN §2).

Unlike the L1 ring (which overwrites and lets lapped readers detect loss), audio
is strict FIFO: the producer BLOCKS when the ring is full (bounded backpressure)
and the consumer never waits — a shortfall is zero-filled and counted, because
an output callback that blocks *is* the underrun.

Priming (the 先読み headroom of DESIGN §2): after construction — and again after
every underrun — the consumer is fed silence until ``prebuffer`` samples are
queued. Start-of-stream priming silence is not an underrun; a *primed* ring that
cannot fill a whole callback IS one (``n_underruns`` += 1), and the ring drops
back to priming to rebuild headroom instead of starving every callback by one
sample forever. The audible gap this inserts is M2b click policy (a): accept the
click, keep the stream alive (smoothing is an M2c+ item).

End-of-stream note: if the producer stops for good while the ring is unprimed,
a tail smaller than ``prebuffer`` stays gated forever. Irrelevant for continuous
live monitoring (the producer never ends — DESIGN §1) and for offline drains
with ``prebuffer=0``; it only trims ≤ prebuffer samples at teardown.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np


class SpscAudioRing:
    """Bounded FIFO of float32 samples with priming / underrun accounting.

    All counters are plain ints updated under the lock; reading them without
    the lock (status displays) is safe under the GIL and only ever one tick
    stale. Lock hold times are a small memcpy — far below the ~10 ms output
    callback budget (same argument as the L1 ring, M0-measured).
    """

    def __init__(self, capacity: int, prebuffer: int):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not 0 <= prebuffer <= capacity:
            raise ValueError("prebuffer must be in [0, capacity]")
        self._buf = np.zeros(int(capacity), dtype=np.float32)
        self._cap = int(capacity)
        self._prebuffer = int(prebuffer)
        self._read = 0    # absolute samples consumed
        self._write = 0   # absolute samples produced
        self._primed = False
        self.n_pushed = 0
        self.n_popped_real = 0   # samples delivered from the ring
        self.n_popped_zero = 0   # zero-filled samples (priming + underrun shortfall)
        self.n_underruns = 0     # primed-but-short pop events
        self._lock = threading.Lock()
        self._space = threading.Condition(self._lock)

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def prebuffer(self) -> int:
        return self._prebuffer

    @property
    def occupancy(self) -> int:
        with self._lock:
            return self._write - self._read

    @property
    def is_primed(self) -> bool:
        with self._lock:
            return self._primed

    # ── producer side (audification worker) ────────────────────────────────
    def push(
        self,
        samples: np.ndarray,
        *,
        stop_event: Optional[threading.Event] = None,
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Append samples, waiting for space (bounded backpressure).

        Returns False if ``stop_event`` is set or ``timeout_s`` elapses before
        everything fits; samples already written by then stay in the ring
        (FIFO order is never violated). Producer-side only.
        """
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return True
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        pos = 0
        with self._space:
            while pos < x.size:
                free = self._cap - (self._write - self._read)
                if free == 0:
                    if stop_event is not None and stop_event.is_set():
                        return False
                    if deadline is not None and time.monotonic() >= deadline:
                        return False
                    self._space.wait(0.05)
                    continue
                n = min(free, x.size - pos)
                w = self._write % self._cap
                end = w + n
                if end <= self._cap:
                    self._buf[w:end] = x[pos:pos + n]
                else:
                    first = self._cap - w
                    self._buf[w:] = x[pos:pos + first]
                    self._buf[:n - first] = x[pos + first:pos + n]
                self._write += n
                self.n_pushed += n
                pos += n
        return True

    # ── consumer side (output callback / sim consumer) ──────────────────────
    def pop_into(self, out: np.ndarray) -> int:
        """Fill ``out`` (1-D float32 view) from the ring; zero-fill any shortfall.

        Never blocks — safe inside the audio output callback (copy + counter
        bumps only). Returns the number of real (non-zero-filled) samples.
        """
        n = out.shape[0]
        if n == 0:
            return 0
        with self._lock:
            occ = self._write - self._read
            if not self._primed:
                if occ >= self._prebuffer:
                    self._primed = True
                else:
                    out[:] = 0.0
                    self.n_popped_zero += n
                    return 0
            take = min(occ, n)
            r = self._read % self._cap
            end = r + take
            if end <= self._cap:
                out[:take] = self._buf[r:end]
            else:
                first = self._cap - r
                out[:first] = self._buf[r:]
                out[first:take] = self._buf[:take - first]
            self._read += take
            self.n_popped_real += take
            if take < n:
                out[take:] = 0.0
                self.n_popped_zero += n - take
                self.n_underruns += 1
                self._primed = False  # rebuild headroom (click policy (a))
            self._space.notify()
        return take
