"""M4 Detector / AdaptiveSnrDetector — adaptive-SNR event detection (DESIGN §4.4).

Numeric, Sim-only (the work order prioritises Sim-number proof of two things):
  * REPELLER-IGNORE: a steady tone (the room's ~25 kHz pest-repeller) is absorbed
    into the adaptive background and NOT detected, while a newly risen sound IS —
    and detection is unaffected by whether the repeller is on or off.
  * MEASUREMENT ACCURACY: peak frequency, duration, bandwidth and chirp slope of a
    known synthetic event come out right.

Signals go through the real ``StftStream`` (the worker's exact data path), then
its dBFS columns into the detector — so these pin the end-to-end DSP, not a mock.
"""

import numpy as np

from ultrascan.detect.detector import AdaptiveSnrDetector, Detector
from ultrascan.detect.events import Event
from ultrascan.dsp.stft import StftStream

FS = 250_000.0
NFFT = 2048
HOP = 1024
BIN_HZ = FS / NFFT          # 122.07 Hz
REPELLER_HZ = 25_000.0


def _make_detector(**kw):
    stft = StftStream(FS, nfft=NFFT, hop=HOP)
    return stft, AdaptiveSnrDetector(stft.freqs_hz, stft.columns_per_second, **kw)


def _signal(dur_s, *, repeller_amp=0.0, bursts=(), noise_amp=2e-3, seed=0):
    """Repeller tone + optional bursts (b0,b1,f,amp) + white noise, all float32."""
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sig = repeller_amp * np.sin(2 * np.pi * REPELLER_HZ * t) + noise_amp * rng.standard_normal(n)
    for b0, b1, bf, ba in bursts:
        i0, i1 = int(b0 * FS), int(b1 * FS)
        sig[i0:i1] += ba * np.sin(2 * np.pi * bf * t[i0:i1])
    return sig.astype(np.float32)


def _chirp(dur_s, b0, b1, f0, f1, *, amp=0.3, noise_amp=2e-3, seed=1):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sig = noise_amp * rng.standard_normal(n)
    i0, i1 = int(b0 * FS), int(b1 * FS)
    tb = t[i0:i1] - b0
    k = (f1 - f0) / (b1 - b0)
    sig[i0:i1] += amp * np.sin(2 * np.pi * (f0 * tb + 0.5 * k * tb ** 2))
    return sig.astype(np.float32)


def _run(sig, **kw):
    stft, det = _make_detector(**kw)
    return det.detect(stft.push(sig)) + det.finalize()


# ── protocol / construction ──────────────────────────────────────────────────
def test_adaptive_snr_satisfies_detector_protocol():
    _, det = _make_detector()
    assert isinstance(det, Detector)


def test_constructor_rejects_bad_params():
    freqs = np.linspace(0, FS / 2, NFFT // 2 + 1)
    for kwargs in (
        dict(columns_per_second=0.0),
        dict(columns_per_second=244.0, snr_threshold_db=0.0),
        dict(columns_per_second=244.0, min_run_bins=0),
        dict(columns_per_second=244.0, meas_db_below_peak=0.0),
        dict(columns_per_second=244.0, bg_tau_s=-1.0),
    ):
        cps = kwargs.pop("columns_per_second")
        try:
            AdaptiveSnrDetector(freqs, cps, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs!r}")


def test_empty_and_noise_only_yield_no_events():
    _, det = _make_detector()
    assert det.detect(np.empty((0, det.n_bins), dtype=np.float32)) == []
    # pure noise floor must not self-trigger (statistical spikes are scattered)
    assert _run(_signal(2.5, noise_amp=2e-3, seed=11)) == []


# ── REPELLER-IGNORE (the heart of method B) ──────────────────────────────────
def test_steady_repeller_is_ignored():
    """A 25 kHz tone present from the start (+ noise) yields NO events: the
    adaptive background absorbs the steady line. Checked across noise seeds."""
    for seed, amp in ((0, 2e-3), (7, 3e-3), (5, 1e-3)):
        ev = _run(_signal(2.5, repeller_amp=0.3, noise_amp=amp, seed=seed))
        assert ev == [], f"steady repeller produced {len(ev)} event(s) (seed {seed})"


def test_burst_detected_over_steady_repeller():
    """The repeller is ignored but a 50 kHz burst that rises mid-stream IS caught."""
    ev = _run(_signal(2.5, repeller_amp=0.3, bursts=((1.0, 1.5, 50_000.0, 0.3),)))
    assert len(ev) == 1
    e = ev[0]
    assert abs(e.f_peak - 50_000.0) < 1.5 * BIN_HZ          # right frequency
    assert 0.95 < e.t_start < 1.05                          # right onset
    assert not (e.f_min <= REPELLER_HZ <= e.f_max)          # NOT the 25 kHz line


def test_detection_is_unaffected_by_repeller_on_off():
    """'Works whether the repeller is on or off': the same burst measures the same
    with the repeller present and absent."""
    on = _run(_signal(2.5, repeller_amp=0.3, bursts=((1.0, 1.5, 50_000.0, 0.3),)))
    off = _run(_signal(2.5, repeller_amp=0.0, bursts=((1.0, 1.5, 50_000.0, 0.3),)))
    assert len(on) == 1 and len(off) == 1
    assert abs(on[0].f_peak - off[0].f_peak) < BIN_HZ
    assert abs((on[0].t_end - on[0].t_start) - (off[0].t_end - off[0].t_start)) < 0.02


# ── MEASUREMENT ACCURACY ─────────────────────────────────────────────────────
def test_measures_tone_frequency_duration_bandwidth():
    ev = _run(_signal(2.5, bursts=((1.0, 1.5, 50_000.0, 0.3),)))
    assert len(ev) == 1
    e = ev[0]
    assert abs(e.f_peak - 50_000.0) < 1.5 * BIN_HZ          # peak freq within ~1 bin
    assert abs((e.t_end - e.t_start) - 0.5) < 0.06          # duration ~0.5 s
    assert e.bandwidth < 1500.0                             # a tone is narrow
    assert e.snr_db > 40.0                                  # well above the floor
    assert e.n_pulses == 1 and e.ipi is None
    assert isinstance(e, Event)


def test_measures_chirp_slope_and_wide_bandwidth():
    """40->60 kHz over 0.5 s: true slope = 20 kHz / 500 ms = 0.040 kHz/ms."""
    ev = _run(_chirp(2.5, 1.0, 1.5, 40_000.0, 60_000.0))
    assert len(ev) == 1
    e = ev[0]
    assert 0.030 < e.slope < 0.050                          # swept up at ~0.04 kHz/ms
    assert e.bandwidth > 15_000.0                           # wide (spans the sweep)
    assert 39_000.0 < e.f_min < 42_000.0
    assert 58_000.0 < e.f_max < 62_000.0


def test_two_sequential_events_are_separated():
    ev = _run(_signal(2.5, bursts=((0.8, 1.1, 40_000.0, 0.2), (1.6, 1.9, 70_000.0, 0.2))))
    assert len(ev) == 2
    ev.sort(key=lambda e: e.t_start)
    assert abs(ev[0].f_peak - 40_000.0) < 1.5 * BIN_HZ
    assert abs(ev[1].f_peak - 70_000.0) < 1.5 * BIN_HZ
    assert ev[0].t_end < ev[1].t_start                      # disjoint in time


def test_short_blip_below_min_event_is_rejected():
    """A 15 ms burst (< the 50 ms min-event floor) is discarded as noise-grade."""
    ev = _run(_signal(2.5, bursts=((1.0, 1.015, 50_000.0, 0.3),)))
    assert ev == []


# ── streaming state: reset / finalize / t_now ────────────────────────────────
def test_reset_clears_state_keeps_time_cursor():
    stft, det = _make_detector()
    det.detect(stft.push(_signal(1.0, repeller_amp=0.3)))
    t_before = det.t_now
    assert det.background_db is not None and t_before > 0.0
    det.reset()
    assert det.background_db is None                        # background dropped
    assert det.t_now == t_before                            # but time preserved


def test_finalize_closes_event_open_at_stream_end():
    """A tone still sounding at end-of-stream is only emitted by finalize()."""
    stft, det = _make_detector()
    # tone from 1.0 s to the very end (no trailing silence to close it)
    sig = _signal(1.8, bursts=((1.0, 1.8, 60_000.0, 0.3),))
    mid = det.detect(stft.push(sig))
    tail = det.finalize()
    assert mid == []                                        # never closed mid-stream
    assert len(tail) == 1
    assert abs(tail[0].f_peak - 60_000.0) < 1.5 * BIN_HZ


def test_t_now_tracks_columns_processed():
    stft, det = _make_detector()
    cols = stft.push(_signal(1.0))
    det.detect(cols)
    assert abs(det.t_now - cols.shape[0] / det.cps) < 1e-9
