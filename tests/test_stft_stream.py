"""M1 tests: streaming display STFT (continuity across block boundaries)."""

import numpy as np

from ultrascan.capture.sources import synth_signal
from ultrascan.dsp.stft import StftStream

FS = 250_000.0


def test_tone_peaks_at_expected_bin():
    stft = StftStream(FS, nfft=2048, hop=1024)
    sig = synth_signal(40_960, FS, tone_hz=45_000.0)
    cols = stft.push(sig)
    assert cols.shape[1] == stft.n_bins == 1025
    assert cols.shape[0] == 1 + (40_960 - 2048) // 1024
    peak_bins = np.argmax(cols, axis=1)
    f_peak = stft.freqs_hz[peak_bins]
    assert np.all(np.abs(f_peak - 45_000.0) < stft.rate / stft.nfft * 2)


def test_full_scale_sine_is_near_zero_dbfs():
    stft = StftStream(FS, nfft=2048, hop=2048)
    sig = synth_signal(20_480, FS, tone_hz=45_000.0, amplitude=1.0)
    cols = stft.push(sig)
    assert abs(float(cols.max())) < 1.0  # calibrated: full-scale tone ~ 0 dBFS


def test_chunked_push_equals_single_push():
    """Framing must be identical no matter how the samples are sliced."""
    sig = synth_signal(50_000, FS, kind="chirp")
    whole = StftStream(FS, nfft=2048, hop=512).push(sig)

    chunked = StftStream(FS, nfft=2048, hop=512)
    pieces, pos = [], 0
    for size in (1, 100, 2047, 2048, 5000, 11111, 29693):
        pieces.append(chunked.push(sig[pos:pos + size]))
        pos += size
    assert pos == sig.size
    joined = np.concatenate([p for p in pieces if p.size], axis=0)
    assert joined.shape == whole.shape
    assert np.allclose(joined, whole, atol=1e-5)


def test_reset_matches_fresh_instance():
    sig = synth_signal(10_000, FS, tone_hz=30_000.0)
    a = StftStream(FS, nfft=1024, hop=512)
    a.push(synth_signal(3_000, FS, tone_hz=80_000.0))  # pollute the carry
    a.reset()
    fresh = StftStream(FS, nfft=1024, hop=512)
    assert np.array_equal(a.push(sig), fresh.push(sig))


def test_underfull_push_buffers_until_one_window():
    stft = StftStream(FS, nfft=2048, hop=1024)
    assert stft.push(synth_signal(1_000, FS)).shape == (0, stft.n_bins)
    assert stft.push(synth_signal(1_000, FS, start_index=1_000)).shape == (0, stft.n_bins)
    cols = stft.push(synth_signal(1_000, FS, start_index=2_000))
    assert cols.shape[0] == 1  # 3000 samples buffered -> exactly one 2048 window
