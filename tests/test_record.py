"""M5 event-triggered GUANO recording (DESIGN §4.5 / §6 M5).

Numeric, Sim-only. The work order's two priorities:
  * PRE-TRIGGER: the recorded WAV must contain the event ONSET plus a pre-roll
    lead-in (read back from L1) — the onset is NOT clipped even though the
    detector only fires a little after the sound begins.
  * GUANO: the WAV reads back at the right sample rate with correct GUANO
    metadata (Timestamp / Samplerate / Make / Model / Length).
Plus: recording is EVENT-ONLY (a steady repeller never triggers a write).
"""

import time
from datetime import datetime

import numpy as np
from scipy.io import wavfile

from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.detect.detector import AdaptiveSnrDetector
from ultrascan.detect.worker import DetectorWorker
from ultrascan.dsp.stft import StftStream
from ultrascan.record.guano_writer import write_event_wav
from ultrascan.record.recorder import EventRecorder

FS = 250_000


def _wait_until(pred, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def _scene(dur_s, *, repeller_amp=0.0, bursts=(), noise_amp=2e-3, seed=0):
    n = int(dur_s * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    sig = repeller_amp * np.sin(2 * np.pi * 25_000.0 * t) + noise_amp * rng.standard_normal(n)
    for b0, b1, bf, ba in bursts:
        i0, i1 = int(b0 * FS), int(b1 * FS)
        sig[i0:i1] += ba * np.sin(2 * np.pi * bf * t[i0:i1])
    return sig.astype(np.float32)


def _run_with_recorder(sig, tmp_dir, *, preroll_s=0.3, postroll_s=0.2):
    l1 = RingBuffer(sig.size)
    l1.write(sig)
    stft = StftStream(FS, nfft=2048, hop=1024)
    det = AdaptiveSnrDetector(stft.freqs_hz, stft.columns_per_second)
    start_abs = l1.reader(from_now=False).position           # == 0 for a freshly filled ring
    rec = EventRecorder(l1, FS, start_abs, preroll_s=preroll_s, postroll_s=postroll_s,
                        out_dir=str(tmp_dir), now=lambda: datetime(2026, 6, 17, 1, 2, 3, 456_000))
    worker = DetectorWorker(l1.reader(from_now=False), stft, det, event_sink=rec)
    worker.start()
    assert _wait_until(lambda: worker.reader.lag == 0)
    worker.stop()
    return rec


# ── write_event_wav (the frozen §4.5 contract) ───────────────────────────────
def test_write_event_wav_roundtrip(tmp_path):
    t = np.arange(int(0.2 * FS)) / FS
    sig = (0.4 * np.sin(2 * np.pi * 50_000.0 * t)).astype(np.float32)
    path = tmp_path / "ev.wav"
    meta = {"Timestamp": datetime(2026, 6, 17, 12, 0, 0), "Make": "Dodotronic",
            "Model": "UltraMic 250K"}
    write_event_wav(sig, FS, meta, str(path))

    rate, data = wavfile.read(str(path))
    assert rate == FS                                   # full 250 kHz
    assert data.dtype == np.int16                       # standard bat-WAV depth
    assert abs(data.size / rate - 0.2) < 1e-3           # ~0.2 s
    spec = np.abs(np.fft.rfft(data.astype(np.float64) * np.hanning(data.size)))
    assert abs(np.argmax(spec) * rate / data.size - 50_000.0) < 200.0   # 50 kHz tone

    from guano import GuanoFile

    gf = GuanoFile(str(path))
    assert gf["Samplerate"] == FS
    assert gf["Make"] == "Dodotronic" and gf["Model"] == "UltraMic 250K"
    assert isinstance(gf["Timestamp"], datetime)
    assert abs(gf["Length"] - 0.2) < 1e-3


def test_write_event_wav_rejects_empty(tmp_path):
    try:
        write_event_wav(np.empty(0, dtype=np.float32), FS, {}, str(tmp_path / "x.wav"))
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty samples")


# ── RingBuffer.read_absolute (pre-roll random access) ────────────────────────
def test_read_absolute_clamps_to_resident():
    ring = RingBuffer(1000)
    ring.write(np.arange(1500, dtype=np.float32))        # resident = abs [500, 1500)
    s, actual = ring.read_absolute(600, 100)             # fully resident
    assert actual == 600 and s.size == 100 and s[0] == 600.0 and s[-1] == 699.0
    s, actual = ring.read_absolute(400, 300)             # front overwritten -> clamp to 500
    assert actual == 500 and s.size == 200 and s[0] == 500.0
    s, actual = ring.read_absolute(1400, 300)            # tail past write head -> clamp to 1500
    assert actual == 1400 and s.size == 100 and s[-1] == 1499.0
    s, _ = ring.read_absolute(0, 100)                    # wholly overwritten -> empty
    assert s.size == 0


# ── PRE-TRIGGER: the recorded onset has a pre-roll lead-in ───────────────────
def test_pretrigger_keeps_onset_with_preroll(tmp_path):
    # noise floor for 1.0 s, then a 50 kHz burst [1.0, 1.5] s, then noise to 2.0 s.
    sig = _scene(2.0, bursts=((1.0, 1.5, 50_000.0, 0.3),))
    preroll_s = 0.3
    rec = _run_with_recorder(sig, tmp_path, preroll_s=preroll_s, postroll_s=0.2)

    assert rec.n_written == 1
    rate, data = wavfile.read(str(rec.last_path))
    assert rate == FS

    # Locate the burst onset INSIDE the recording via a smoothed amplitude envelope.
    env = np.convolve(np.abs(data.astype(np.float64)), np.ones(256) / 256, mode="same")
    onset_idx = int(np.argmax(env > 0.2 * env.max()))
    onset_s = onset_idx / FS
    # The onset must sit ~preroll INTO the file (lead-in present, onset not clipped).
    assert 0.2 < onset_s < 0.45, f"onset at {onset_s * 1e3:.0f} ms (preroll {preroll_s * 1e3:.0f} ms)"
    # File length ~ preroll + event + postroll = 1.0 s.
    assert abs(data.size / FS - 1.0) < 0.1
    assert rec.last_lead_s > 0.2          # pre-roll actually captured


# ── EVENT-ONLY: steady repeller never triggers a recording ───────────────────
def test_records_event_only_not_repeller(tmp_path):
    only_repeller = _scene(2.5, repeller_amp=0.3)
    rec = _run_with_recorder(only_repeller, tmp_path / "a")
    assert rec.n_written == 0                            # repeller absorbed -> nothing written

    with_burst = _scene(2.5, repeller_amp=0.3, bursts=((1.0, 1.5, 50_000.0, 0.3),))
    rec2 = _run_with_recorder(with_burst, tmp_path / "b")
    assert rec2.n_written == 1                           # only the burst is recorded
    assert rec2.last_path.exists()
