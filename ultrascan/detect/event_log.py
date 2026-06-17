"""CSV logger for detected events — the M4 MEASUREMENT log.

This logs event *measurements* (frequency / duration / bandwidth / SNR …), NOT
audio. High-rate WAV event recording is M5 (``record/guano_writer.py``). Off by
default at the CLI; enabled with ``--log-events PATH``. One header + one row per
completed event, flushed per row so a crash keeps the detections already seen.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from ultrascan.detect.events import Event

FIELDS = [
    "wall_clock", "t_start_s", "t_end_s", "duration_s", "f_peak_hz",
    "f_min_hz", "f_max_hz", "bandwidth_hz", "slope_khz_per_ms",
    "n_pulses", "ipi_s", "snr_db",
]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).isoformat(timespec="milliseconds")


class EventCsvLogger:
    """Append-per-event CSV sink; use as a ``DetectorWorker`` ``event_sink``.

    ``clock`` is injectable for deterministic tests (defaults to wall-clock)."""

    def __init__(self, path, *, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(FIELDS)
        self._fh.flush()
        self.n_written = 0

    def __call__(self, ev: Event) -> None:
        self._w.writerow([
            _iso(self._clock()),
            f"{ev.t_start:.4f}", f"{ev.t_end:.4f}", f"{ev.t_end - ev.t_start:.4f}",
            f"{ev.f_peak:.1f}", f"{ev.f_min:.1f}", f"{ev.f_max:.1f}",
            f"{ev.bandwidth:.1f}", f"{ev.slope:.5f}",
            ev.n_pulses, "" if ev.ipi is None else f"{ev.ipi:.4f}",
            f"{ev.snr_db:.1f}",
        ])
        self._fh.flush()           # durable per event (crash-safe log)
        self.n_written += 1

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
