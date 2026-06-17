"""L5 event recording — pre-roll grab + GUANO WAV write (DESIGN §6 M5).

An ``EventRecorder`` is used as a ``DetectorWorker`` ``event_sink``: on each
completed event it slices the raw L1 samples for
``[onset - preroll, end + postroll]`` and writes a high-rate GUANO WAV.

PRE-TRIGGER (the point of M5): the detector only flags an event a little AFTER it
begins, so naively recording from "now" would clip the onset. Instead we read
BACK from L1 — the truth source doubles as the pre-trigger ring buffer
(``RingBuffer.read_absolute``) — starting ``preroll`` before the onset. An event's
detector-relative time maps to an absolute L1 sample as
``l1_start_abs + t * fs`` (exact when the detector keeps up; see the drop note).

Thread (DESIGN §2): called on the detector WORKER thread, never an audio
callback — disk I/O stays off the realtime path. Recording is EVENT-ONLY: a
steady tone / silence is absorbed by the detector and never triggers a write.

Drop note: the time->sample mapping assumes the detector read L1 without gaps
(the normal case; all Sim tests). Under sustained overload (an L1 reader gap →
the worker resets the detector), the window for subsequent events is offset by
the dropped-sample count — the recording still happens, just mis-aligned. Real
events are short and the detector is light, so gaps are not expected.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.detect.events import Event
from ultrascan.record.guano_writer import write_event_wav

MAKE = "Dodotronic"
MODEL = "UltraMic 250K"


class EventRecorder:
    """Detector ``event_sink`` that writes a pre-rolled GUANO WAV per event.

    Parameters
    ----------
    ring : RingBuffer
        The L1 ring (read back for the pre-roll). Must be long enough to still
        hold the onset when the event closes (event_dur + preroll < ring seconds).
    fs : float
        Capture sample rate (Hz) — the WAV is written at this full rate.
    l1_start_abs : int
        Absolute L1 index the detector began at (``DetectorWorker.l1_start_abs``).
    preroll_s, postroll_s : float
        Lead-in before the onset / tail after the end to capture.
    max_record_s : float
        Hard cap on a single recording's length (bounds the L1 copy under lock).
    out_dir : str
        Directory for the WAVs (date/time-stamped names; BTO-compatible).
    now : callable() -> datetime
        Injectable clock (Timestamp + filename); defaults to ``datetime.now``.
    """

    def __init__(
        self,
        ring: RingBuffer,
        fs: float,
        l1_start_abs: int,
        *,
        preroll_s: float = 0.3,
        postroll_s: float = 0.2,
        max_record_s: float = 10.0,
        out_dir: str = "captures",
        make: str = MAKE,
        model: str = MODEL,
        now: Callable[[], datetime] = datetime.now,
    ):
        if preroll_s < 0 or postroll_s < 0:
            raise ValueError("preroll_s / postroll_s must be >= 0")
        if max_record_s <= 0:
            raise ValueError("max_record_s must be > 0")
        self.ring = ring
        self.fs = int(fs)
        self.l1_start_abs = int(l1_start_abs)
        self.preroll = int(round(preroll_s * fs))
        self.postroll = int(round(postroll_s * fs))
        self.max_record = int(round(max_record_s * fs))
        self.out_dir = Path(out_dir)
        self.make = make
        self.model = model
        self._now = now
        self.n_written = 0
        self.last_path: Optional[Path] = None
        self.last_lead_s = 0.0

    def __call__(self, event: Event) -> Optional[Path]:
        abs_onset = self.l1_start_abs + int(round(event.t_start * self.fs))
        abs_end = self.l1_start_abs + int(round(event.t_end * self.fs))
        start = abs_onset - self.preroll
        n = min((abs_end + self.postroll) - start, self.max_record)
        samples, actual_start = self.ring.read_absolute(start, n)
        if samples.size == 0:
            return None  # the window has already been overwritten (event too old)

        lead_s = max(0, abs_onset - actual_start) / self.fs   # pre-roll actually captured
        ts = self._now()
        path = self.out_dir / self._filename(ts, event)
        meta = {
            "Timestamp": ts,
            "Make": self.make,
            "Model": self.model,
            "Note": (
                f"ultrascan event: f_peak={event.f_peak / 1e3:.1f}kHz "
                f"f=[{event.f_min / 1e3:.1f},{event.f_max / 1e3:.1f}]kHz "
                f"snr={event.snr_db:.0f}dB preroll={lead_s * 1e3:.0f}ms"
            ),
        }
        write_event_wav(samples, self.fs, meta, str(path))
        self.n_written += 1
        self.last_path = path
        self.last_lead_s = lead_s
        return path

    def _filename(self, ts: datetime, event: Event) -> str:
        # BTO-compatible: starts with date/time; ms + peak-kHz keep it unique/informative.
        return (
            f"{ts:%Y%m%d_%H%M%S}_{ts.microsecond // 1000:03d}_"
            f"{event.f_peak / 1e3:.0f}kHz.wav"
        )
