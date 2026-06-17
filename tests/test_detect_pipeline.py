"""M4 plumbing: DetectorWorker (L1 -> own STFT -> detector) + EventCsvLogger.

The detector core is pinned numerically in test_detector.py; here we check the
worker wiring (an independent L1 consumer that finds the burst and ignores the
repeller) and the CSV measurement log.
"""

import time

import numpy as np

from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.detect.detector import AdaptiveSnrDetector
from ultrascan.detect.event_log import FIELDS, EventCsvLogger
from ultrascan.detect.events import Event
from ultrascan.detect.worker import DetectorWorker
from ultrascan.dsp.stft import StftStream

FS = 250_000.0


def _wait_until(pred, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _scene(dur_s, *, repeller_amp=0.3, burst=(1.0, 1.5, 50_000.0, 0.3), noise_amp=2e-3, seed=0):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sig = repeller_amp * np.sin(2 * np.pi * 25_000.0 * t) + noise_amp * rng.standard_normal(n)
    if burst:
        b0, b1, bf, ba = burst
        i0, i1 = int(b0 * FS), int(b1 * FS)
        sig[i0:i1] += ba * np.sin(2 * np.pi * bf * t[i0:i1])
    return sig.astype(np.float32)


def _run_worker(sig, **kw):
    l1 = RingBuffer(sig.size)
    l1.write(sig)
    stft = StftStream(FS, nfft=2048, hop=1024)
    det = AdaptiveSnrDetector(stft.freqs_hz, stft.columns_per_second, **kw)
    seen = []
    worker = DetectorWorker(l1.reader(from_now=False), stft, det, event_sink=seen.append)
    worker.start()
    assert _wait_until(lambda: worker.reader.lag == 0)
    worker.stop()      # joins + finalizes any open event
    return worker, seen


def test_detector_worker_finds_burst_ignores_repeller():
    worker, seen = _run_worker(_scene(2.5))
    assert worker.n_events == 1
    e = seen[0]
    assert abs(e.f_peak - 50_000.0) < 300.0          # the burst
    assert not (e.f_min <= 25_000.0 <= e.f_max)       # NOT the repeller line
    events, t_now = worker.snapshot()
    assert len(events) == 1 and t_now > 2.0


def test_detector_worker_repeller_only_is_silent():
    worker, seen = _run_worker(_scene(2.5, burst=None))
    assert worker.n_events == 0
    assert seen == []


def test_event_csv_logger_roundtrip(tmp_path):
    path = tmp_path / "events.csv"
    log = EventCsvLogger(path, clock=lambda: 1_700_000_000.0)
    ev = Event(t_start=1.0, t_end=1.5, f_peak=50_000.0, f_min=49_800.0, f_max=50_200.0,
               bandwidth=400.0, slope=0.04, n_pulses=1, ipi=None, snr_db=42.5)
    log(ev)
    log(ev)
    log.close()

    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split(",") == FIELDS               # header
    assert len(rows) == 3                              # header + 2 events
    cells = dict(zip(FIELDS, rows[1].split(",")))
    assert float(cells["f_peak_hz"]) == 50_000.0
    assert float(cells["duration_s"]) == 0.5
    assert float(cells["slope_khz_per_ms"]) == 0.04
    assert cells["ipi_s"] == ""                        # None -> empty cell
    assert log.n_written == 2
