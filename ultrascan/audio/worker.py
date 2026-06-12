"""L3 audification worker (DESIGN §3): own L1 cursor -> Audifier -> SPSC ring.

This is a *separate* L1 consumer from the display ``DspWorker`` (DESIGN §1
two-pass separation): audio is strict FIFO and shares no queue with the display
path. The frozen ``HeterodyneAudifier`` is driven exclusively from this thread,
so its stateful ``process()``/``configure()`` never race.

Band re-selection is asynchronous: any thread (the GUI) calls
``request_band()``; the worker applies it between ``process()`` calls.
``configure()`` is atomic (M2a), so a refused request leaves the running stream
intact — the error string is exposed for the status bar instead of raised.

Click policy (a) — decided for M2b: ``configure()`` fully resets DDC state, so
a band re-selection produces an audible click; an input gap (lapped L1 reader,
``n_dropped_in``) likewise passes through as a transient. Both are ACCEPTED for
M2b — continuous sound has priority; crossfade/ramp smoothing is recorded as an
M2c+ item, deliberately not implemented here.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

from ultrascan.capture.ring_buffer import RingReader
from ultrascan.audio.spsc import SpscAudioRing


class AudioWorker(threading.Thread):
    """Pull raw samples from the L1 ring, audify, push to the SPSC ring."""

    def __init__(
        self,
        reader: RingReader,
        audifier,
        out_ring: SpscAudioRing,
        fs_in: float,
        f_lo_sel: float,
        bandwidth: float,
        *,
        volume: float = 1.0,
        poll_s: float = 0.01,
        max_read: Optional[int] = None,
    ):
        super().__init__(name="audio-worker", daemon=True)
        self.reader = reader
        self.audifier = audifier
        self.out_ring = out_ring
        self.fs_in = float(fs_in)
        # Fixed output attenuator — a safety knob for live listening, applied on
        # this thread (never in the callback). NOT the M3 GainStage (no AGC /
        # normalize / compress lives here).
        self.volume = float(volume)
        self.poll_s = float(poll_s)
        # Bound one read so a backlog cannot freeze a poll cycle (~0.25 s default).
        self.max_read = int(max_read) if max_read else int(0.25 * self.fs_in)

        # Initial configure runs on the constructing thread, before start() —
        # no concurrency yet. Invalid CLI selections raise here, loudly.
        self.audifier.configure(f_lo_sel, bandwidth, self.fs_in)
        self.band: Tuple[float, float] = (float(f_lo_sel), float(bandwidth))
        self.last_band_error: Optional[str] = None

        self.n_in_samples = 0
        self.n_out_samples = 0
        self.n_dropped_in = 0    # L1 gap passed through as a transient (policy (a))
        self.n_band_changes = 0
        self.n_push_failed = 0   # samples abandoned because push() hit stop/timeout

        self._pending_band: Optional[Tuple[float, float]] = None
        self._band_lock = threading.Lock()
        # NB: not `_stop` — that shadows threading.Thread._stop() (M1 finding).
        self._stop_evt = threading.Event()

    # ── any thread ──────────────────────────────────────────────────────────
    def request_band(self, f_lo_sel: float, bandwidth: float) -> None:
        """Queue a band re-selection; the worker applies it between blocks."""
        with self._band_lock:
            self._pending_band = (float(f_lo_sel), float(bandwidth))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=timeout)

    # ── worker thread only ──────────────────────────────────────────────────
    def _apply_pending_band(self) -> None:
        with self._band_lock:
            req, self._pending_band = self._pending_band, None
        if req is None:
            return
        try:
            # Full state reset -> audible click at the boundary (policy (a)).
            self.audifier.configure(req[0], req[1], self.fs_in)
            self.band = req
            self.last_band_error = None
            self.n_band_changes += 1
        except ValueError as exc:
            # Atomic configure refused the selection: old band keeps playing.
            self.last_band_error = str(exc)

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._apply_pending_band()
            samples, dropped = self.reader.read_new(max_samples=self.max_read)
            if dropped:
                self.n_dropped_in += dropped
            if samples.size:
                self.n_in_samples += samples.size
                audio = self.audifier.process(samples)
                if self.volume != 1.0:
                    audio *= np.float32(self.volume)
                if audio.size:
                    if self.out_ring.push(audio, stop_event=self._stop_evt):
                        self.n_out_samples += audio.size
                    else:
                        self.n_push_failed += audio.size
            if self.reader.lag < self.max_read:  # caught up -> sleep one poll
                self._stop_evt.wait(self.poll_s)
