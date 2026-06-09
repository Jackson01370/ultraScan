"""Sim-first unit tests for the M0 capture sources (no hardware required)."""

import numpy as np
import pytest

from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.capture.sources import (
    SyntheticSource,
    WavFileSource,
    make_source,
    synth_signal,
)


# ── synth_signal ───────────────────────────────────────────────────────────
def test_synth_tone_peaks_at_expected_frequency():
    fs = 250_000.0
    n = 65_536
    sig = synth_signal(n, fs, kind="tone", tone_hz=45_000.0)
    assert sig.dtype == np.float32
    spec = np.abs(np.fft.rfft(sig * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    peak = freqs[int(np.argmax(spec))]
    assert abs(peak - 45_000.0) < 100.0  # bin width ~3.8 Hz


def test_synth_blocks_are_phase_continuous():
    fs = 250_000.0
    a = synth_signal(1024, fs, start_index=0, tone_hz=40_000.0)
    b = synth_signal(1024, fs, start_index=1024, tone_hz=40_000.0)
    joined = np.concatenate([a, b])
    full = synth_signal(2048, fs, start_index=0, tone_hz=40_000.0)
    assert np.allclose(joined, full, atol=1e-5)


# ── RingBuffer ─────────────────────────────────────────────────────────────
def test_ring_buffer_keeps_latest_in_order():
    rb = RingBuffer(8)
    rb.write(np.arange(5, dtype=np.float32))
    rb.write(np.arange(5, 12, dtype=np.float32))  # forces wrap (total 12 > cap 8)
    assert rb.total_written == 12
    assert np.array_equal(rb.latest(8), np.arange(4, 12, dtype=np.float32))
    assert np.array_equal(rb.latest(3), np.arange(9, 12, dtype=np.float32))


def test_ring_buffer_block_larger_than_capacity_keeps_tail():
    rb = RingBuffer(4)
    rb.write(np.arange(10, dtype=np.float32))
    assert np.array_equal(rb.latest(4), np.array([6, 7, 8, 9], dtype=np.float32))


def test_ring_buffer_latest_clamps_to_available():
    rb = RingBuffer(16)
    rb.write(np.arange(3, dtype=np.float32))
    assert rb.latest(10).size == 3


# ── SyntheticSource streaming (short, realtime) ────────────────────────────
def test_synthetic_source_streams_blocks():
    src = SyntheticSource(samplerate=250_000.0, blocksize=2048, tone_hz=45_000.0)
    blocks = []
    overflow = []

    def on_block(block, status):
        blocks.append(block.size)
        overflow.append(status.input_overflow)

    src.start(on_block)
    # ~0.25 s of capture; cheap consumer -> no simulated overruns expected.
    import time

    time.sleep(0.25)
    src.stop()

    assert not src.is_running
    assert len(blocks) > 5
    assert sum(blocks) > 0
    assert not any(overflow)


# ── WavFileSource loader ───────────────────────────────────────────────────
def test_wav_source_loads_and_normalizes(tmp_path):
    from scipy.io import wavfile

    fs = 250_000
    sig = synth_signal(8192, fs, tone_hz=50_000.0, amplitude=1.0)  # full-scale probe
    pcm = (sig * 32767).astype(np.int16)
    path = tmp_path / "tone50k.wav"
    wavfile.write(str(path), fs, pcm)

    src = WavFileSource(str(path), blocksize=1024)
    assert src.samplerate == fs
    assert src.data.dtype == np.float32
    assert src.data.size == 8192
    # int16 -> float normalized into [-1, 1]; this 50 kHz/250 kHz tone peaks at ~0.95
    peak = float(np.max(np.abs(src.data)))
    assert 0.5 < peak <= 1.0


def test_make_source_unknown_raises():
    with pytest.raises(ValueError):
        make_source("bogus")
