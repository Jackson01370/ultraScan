"""Minimal float32 ring buffer — the M0 seed of L1 (the single source of truth).

DESIGN §3 calls for a single-producer / multi-consumer ring where each consumer
holds its own read pointer. That full version lands at M1. For M0 we only need a
fixed-size circular store of the most-recent samples so the stress test can run an
offline FFT after capture, so this is deliberately minimal (one writer, snapshot
reads). It is thread-safe for one producer thread + reader on the main thread.
"""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    def __init__(self, capacity: int, dtype=np.float32):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = int(capacity)
        self._buf = np.zeros(self._capacity, dtype=dtype)
        self._write = 0          # next write index (mod capacity)
        self._total = 0          # total samples ever written
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        return self._total

    def write(self, block: np.ndarray) -> None:
        """Append a 1-D block, overwriting oldest samples on wrap. Copy-only path."""
        block = np.asarray(block).reshape(-1)
        n = block.size
        if n == 0:
            return
        with self._lock:
            if n >= self._capacity:
                # block larger than the ring: keep only its tail
                self._buf[:] = block[-self._capacity:]
                self._write = 0
                self._total += n
                return
            end = self._write + n
            if end <= self._capacity:
                self._buf[self._write:end] = block
            else:
                first = self._capacity - self._write
                self._buf[self._write:] = block[:first]
                self._buf[: n - first] = block[first:]
            self._write = end % self._capacity
            self._total += n

    def latest(self, n: int) -> np.ndarray:
        """Return up to the most-recent ``n`` samples, oldest-first."""
        with self._lock:
            available = min(self._total, self._capacity)
            n = min(int(n), available)
            if n == 0:
                return np.empty(0, dtype=self._buf.dtype)
            start = (self._write - n) % self._capacity
            end = start + n
            if end <= self._capacity:
                return self._buf[start:end].copy()
            first = self._capacity - start
            out = np.empty(n, dtype=self._buf.dtype)
            out[:first] = self._buf[start:]
            out[first:] = self._buf[: n - first]
            return out
