"""L1 ring buffer — the single source of truth (DESIGN §3).

Single producer + multiple consumers; every consumer holds its *own* read
pointer (``RingReader``), so display / audification / recording rates are fully
decoupled. The writer never blocks on readers: the ring overwrites the oldest
samples, and a reader that falls behind detects the loss as a ``dropped`` count
on its next read (display may shrug it off; audio must handle it — DESIGN §1).

The M0 snapshot API (``write`` / ``latest``) is unchanged; M1 adds the
per-consumer cursor reads. All access is serialized by one lock — copies at
250 kHz float32 are ~1 MB/s, so worst-case lock hold time stays far below the
~1 ms capture-callback budget.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

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
                # Block fills (or laps) the whole ring: keep only its tail, placed so
                # the invariant "absolute sample i lives at buf[i % capacity]" holds —
                # RingReader addresses the buffer in absolute coordinates.
                new_total = self._total + n
                start = new_total % self._capacity  # == (new_total - capacity) % capacity
                tail = block[-self._capacity:]
                first = self._capacity - start
                self._buf[start:] = tail[:first]
                self._buf[:start] = tail[first:]
                self._write = start
                self._total = new_total
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

    def reader(self, *, from_now: bool = True) -> "RingReader":
        """Create an independent consumer cursor (DESIGN §3: pointers not shared).

        ``from_now=True`` starts at the current write head (no backlog);
        ``from_now=False`` starts at the oldest sample still held in the ring.
        """
        return RingReader(self, from_now=from_now)

    def _copy_absolute(self, start: int, n: int) -> np.ndarray:
        """Copy ``n`` samples beginning at *absolute* position ``start``.

        Caller must hold ``self._lock`` and guarantee the span is still resident
        (i.e. ``start >= total - capacity`` and ``start + n <= total``).
        """
        s = start % self._capacity
        e = s + n
        if e <= self._capacity:
            return self._buf[s:e].copy()
        out = np.empty(n, dtype=self._buf.dtype)
        first = self._capacity - s
        out[:first] = self._buf[s:]
        out[first:] = self._buf[: n - first]
        return out


class RingReader:
    """One consumer's read cursor over a :class:`RingBuffer`.

    FIFO semantics in absolute sample coordinates: ``read_new`` returns samples
    the cursor has not seen yet, oldest-first. If the producer lapped this
    reader, the overwritten span is reported as ``dropped`` (count of lost
    samples) and the cursor resumes at the oldest sample still resident —
    consumers use that signal to reset stream state (e.g. STFT overlap carry).
    """

    def __init__(self, ring: RingBuffer, *, from_now: bool = True):
        self._ring = ring
        with ring._lock:
            if from_now:
                self._pos = ring._total
            else:
                self._pos = ring._total - min(ring._total, ring._capacity)

    @property
    def position(self) -> int:
        """Absolute sample index this reader will read next."""
        return self._pos

    @property
    def lag(self) -> int:
        """Samples written but not yet read by this consumer (includes lost ones)."""
        with self._ring._lock:
            return self._ring._total - self._pos

    def read_new(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Return ``(samples, dropped)`` — unread samples oldest-first.

        ``dropped`` is the number of samples overwritten before this reader got
        to them (0 when keeping up). ``max_samples`` caps the returned block;
        the remainder stays queued for the next call (strict FIFO).
        """
        ring = self._ring
        with ring._lock:
            total = ring._total
            oldest = total - min(total, ring._capacity)
            dropped = max(0, oldest - self._pos)
            start = self._pos + dropped
            n = total - start
            if max_samples is not None:
                n = min(n, max(0, int(max_samples)))
            if n <= 0:
                self._pos = start
                return np.empty(0, dtype=ring._buf.dtype), dropped
            out = ring._copy_absolute(start, n)
            self._pos = start + n
            return out, dropped

    def skip_to_latest(self) -> int:
        """Jump the cursor to the write head; returns how many samples were skipped."""
        with self._ring._lock:
            skipped = self._ring._total - self._pos
            self._pos = self._ring._total
            return skipped
