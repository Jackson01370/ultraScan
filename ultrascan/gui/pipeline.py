"""Display-path plumbing: L0 callback glue -> L1 ring -> L2 worker -> L4 queue.

Thread rules enforced here (DESIGN §2):
  - ``RingWriter.on_block`` runs on the capture callback: COPY ONLY
    (ring memcpy + counter bumps; no FFT, no GUI, no allocation-heavy work).
  - ``DspWorker`` is a plain thread: reads its own ring cursor, runs the STFT,
    appends finished columns to a bounded deque. It never touches Qt.
  - The GUI drains the deque from a QTimer on the GUI thread. The deque is
    latest-priority: when the GUI stalls, ``maxlen`` silently sheds the oldest
    batches (display may drop frames — DESIGN §1; audio will NOT use this path).
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Optional

import numpy as np

from ultrascan.capture.ring_buffer import RingBuffer, RingReader
from ultrascan.capture.sources import CaptureStatus
from ultrascan.dsp.stft import StftStream


class RingWriter:
    """Capture-callback consumer: copy the block into L1, count Xruns. Nothing else."""

    def __init__(self, ring: RingBuffer):
        self.ring = ring
        self.n_blocks = 0
        self.n_xruns = 0

    def on_block(self, block: np.ndarray, status: CaptureStatus) -> None:
        # COPY ONLY (DESIGN §2) — heavy work happens in DspWorker, off this thread.
        self.ring.write(block)
        self.n_blocks += 1
        if status.input_overflow or status.input_underflow:
            self.n_xruns += 1


class DspWorker(threading.Thread):
    """L2: pull new samples from an own ring cursor, emit STFT columns.

    A gap reported by the reader (``dropped > 0``) invalidates the STFT overlap
    carry, so the stream is reset — one visibly missing waterfall slice instead
    of a smeared, misaligned one.
    """

    def __init__(
        self,
        reader: RingReader,
        stft: StftStream,
        out_queue: "Deque[np.ndarray]",
        poll_s: float = 0.015,
        max_read: Optional[int] = None,
    ):
        super().__init__(name="dsp-worker", daemon=True)
        self.reader = reader
        self.stft = stft
        self.out_queue = out_queue
        self.poll_s = float(poll_s)
        # Bound one read so a huge backlog cannot freeze a poll cycle (~1 s default).
        self.max_read = int(max_read) if max_read else int(stft.rate)
        self.n_columns = 0
        self.n_dropped_samples = 0
        # NB: must not be named `_stop` — that shadows threading.Thread's internal
        # _stop() and breaks join().
        self._stop_evt = threading.Event()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=timeout)

    def run(self) -> None:
        while not self._stop_evt.is_set():
            samples, dropped = self.reader.read_new(max_samples=self.max_read)
            if dropped:
                self.n_dropped_samples += dropped
                self.stft.reset()
            if samples.size:
                cols = self.stft.push(samples)
                if cols.size:
                    self.n_columns += cols.shape[0]
                    self.out_queue.append(cols)
            if self.reader.lag < self.max_read:  # caught up -> sleep one poll
                self._stop_evt.wait(self.poll_s)


def make_display_queue(maxlen: int = 64) -> "Deque[np.ndarray]":
    """Latest-priority column queue (oldest batches shed when the GUI stalls)."""
    return deque(maxlen=maxlen)
