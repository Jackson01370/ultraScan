"""L5 detection worker — a third independent L1 consumer (DESIGN §3 / §6 M4).

Runs the adaptive-SNR detector on its OWN ring cursor + its OWN ``StftStream``,
fully decoupled from the display (``DspWorker``) and audio (``AudioWorker``)
paths. That decoupling is the point: the display deque is latest-priority and may
drop frames, but the detector must see a CONTIGUOUS frame stream to segment
events in time — so it reads L1 directly (its own ``RingReader``) and resets
(STFT carry + detector state) on any reader gap, exactly like ``DspWorker`` does.

Thread rules (DESIGN §2): plain worker thread. It never touches Qt and never
touches the audio callback path — detection runs alongside listening without
perturbing it. The GUI reads :meth:`snapshot` from its own QTimer; an optional
``event_sink`` (e.g. the CSV logger) is called on THIS thread per completed event.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Deque, List, Optional, Tuple

from ultrascan.capture.ring_buffer import RingReader
from ultrascan.detect.detector import AdaptiveSnrDetector
from ultrascan.detect.events import Event
from ultrascan.dsp.stft import StftStream


class DetectorWorker(threading.Thread):
    """Drive ``AdaptiveSnrDetector`` over an L1 cursor; collect events.

    Parameters
    ----------
    reader : RingReader
        This worker's OWN L1 cursor (separate from display / audio cursors).
    stft : StftStream
        This worker's OWN streaming STFT (same nfft/hop as the display by
        convention, so overlay boxes line up — but a distinct instance).
    detector : AdaptiveSnrDetector
        Built from ``stft.freqs_hz`` / ``stft.columns_per_second``.
    history_s : float
        How long completed events are retained for the GUI overlay snapshot.
    event_sink : callable(Event) | None
        Optional per-event callback (e.g. CSV logger), invoked on the worker
        thread as each event completes. Exceptions in it never kill the worker.
    """

    def __init__(
        self,
        reader: RingReader,
        stft: StftStream,
        detector: AdaptiveSnrDetector,
        *,
        history_s: float = 12.0,
        event_sink: Optional[Callable[[Event], None]] = None,
        poll_s: float = 0.02,
        max_read: Optional[int] = None,
    ):
        super().__init__(name="detector-worker", daemon=True)
        self.reader = reader
        # Absolute L1 index where detection began: lets the recorder map an
        # event's detector-relative time to absolute L1 samples (M5 pre-roll).
        self.l1_start_abs = reader.position
        self.stft = stft
        self.detector = detector
        self.event_sink = event_sink
        self.poll_s = float(poll_s)
        self.max_read = int(max_read) if max_read else int(stft.rate)
        self._history_s = float(history_s)
        self.n_events = 0
        self.n_dropped_samples = 0
        self._events: Deque[Event] = deque()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=timeout)
        # Flush an event still open at shutdown (worker thread is now stopped, so
        # the detector is touched single-threadedly here).
        for ev in self.detector.finalize():
            self._emit(ev)

    def run(self) -> None:
        while not self._stop_evt.is_set():
            samples, dropped = self.reader.read_new(max_samples=self.max_read)
            if dropped:
                self.n_dropped_samples += dropped
                self.stft.reset()
                self.detector.reset()
            if samples.size:
                cols = self.stft.push(samples)
                if cols.size:
                    for ev in self.detector.detect(cols):
                        self._emit(ev)
            if self.reader.lag < self.max_read:   # caught up -> sleep one poll
                self._stop_evt.wait(self.poll_s)

    def _emit(self, ev: Event) -> None:
        self.n_events += 1
        cutoff = self.detector.t_now - self._history_s
        with self._lock:
            self._events.append(ev)
            while self._events and self._events[0].t_end < cutoff:
                self._events.popleft()           # bound memory even with no GUI
        if self.event_sink is not None:
            try:
                self.event_sink(ev)
            except Exception:                    # logging must never kill detection
                pass

    def snapshot(self) -> Tuple[List[Event], float]:
        """Thread-safe (events, t_now) for the GUI overlay; prunes scrolled-off."""
        t_now = self.detector.t_now
        cutoff = t_now - self._history_s
        with self._lock:
            while self._events and self._events[0].t_end < cutoff:
                self._events.popleft()
            return list(self._events), t_now
