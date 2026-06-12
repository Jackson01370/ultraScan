"""M2a tests: HeterodyneAudifier DDC chain — numerical correctness, offline.

Pins the frozen §4.2 contract behaviour: image-free complex mix (vs the
PROHIBITED real-cos mix), block-boundary continuity (NCO phase / FIR zi /
decimation phase / soxr state), frequency mapping, and the real-capture WAV.
No audio output here — speakers/output callback are M2b.
"""

from pathlib import Path

import numpy as np
import pytest

from ultrascan.capture.sources import synth_signal
from ultrascan.dsp.audifier import FS_OUT, Audifier, HeterodyneAudifier
from ultrascan.dsp.ddc import ddc_heterodyne

FS = 250_000.0
WAV = Path(__file__).resolve().parent.parent / "captures" / "m0_ultramic_keys_250k.wav"


# ── spectrum helpers (offline analysis) ─────────────────────────────────────
def _spectrum_db(x: np.ndarray, fs: float):
    """Hann + rfft in dB, skipping startup transients (FIR + soxr latency)."""
    x = x[4800:]
    w = np.hanning(x.size)
    mag = np.abs(np.fft.rfft(x * w))
    return np.fft.rfftfreq(x.size, 1.0 / fs), 20.0 * np.log10(mag + 1e-12)


def _band_peak_db(freqs, db, f0, tol=200.0) -> float:
    m = (freqs >= f0 - tol) & (freqs <= f0 + tol)
    return float(db[m].max())


def _tone_out(f_tone, f_lo, bandwidth=10_000.0, dur=1.0) -> np.ndarray:
    aud = HeterodyneAudifier()
    aud.configure(f_lo, bandwidth, FS)
    return aud.process(synth_signal(int(FS * dur), FS, tone_hz=f_tone, amplitude=0.5))


def _cos_mix_reference(f_tone, f_lo, bandwidth=10_000.0, dur=1.0) -> np.ndarray:
    """The PROHIBITED chain (§11): real cos mix + real LPF + decimate + soxr.

    Test-local on purpose — exists only to demonstrate the image that the
    complex mix removes.
    """
    import soxr
    from scipy.signal import firwin, lfilter

    sig = synth_signal(int(FS * dur), FS, tone_hz=f_tone, amplitude=0.5).astype(np.float64)
    n = np.arange(sig.size)
    mixed = sig * np.cos(2.0 * np.pi * f_lo * n / FS) * 2.0
    y = lfilter(firwin(255, min(bandwidth, 25_000.0), fs=FS), 1.0, mixed)
    return soxr.resample(y[::5].astype(np.float32), 50_000, 48_000)


# ── frequency mapping (§5: band lower edge -> DC) ───────────────────────────
def test_mapping_45k_with_lo40k_lands_at_5k():
    out = _tone_out(45_000.0, 40_000.0)
    assert out.dtype == np.float32
    # 1 s in -> ~1 s out at 48k, minus FIR/soxr latency
    assert 46_000 < out.size <= 48_001
    freqs, db = _spectrum_db(out, FS_OUT)
    peak = freqs[int(np.argmax(db))]
    assert abs(peak - 5_000.0) < 50.0


def test_mapping_25k_repeller_band_with_lo20k_lands_at_5k():
    out = _tone_out(25_000.0, 20_000.0)
    freqs, db = _spectrum_db(out, FS_OUT)
    peak = freqs[int(np.argmax(db))]
    assert abs(peak - 5_000.0) < 50.0


def test_passband_amplitude_survives_analytic_halving():
    # Gain-2 taps (review): a 0.5-amplitude tone in the passband must come out
    # at ~0.5 amplitude (RMS 0.5/sqrt(2)), not halved by the analytic-signal step.
    out = _tone_out(45_000.0, 40_000.0)
    seg = out[4800:].astype(np.float64)
    rms = float(np.sqrt(np.mean(seg * seg)))
    assert abs(rms - 0.5 / np.sqrt(2.0)) < 0.01


# ── mirror test (§9): complex mix rejects the image, cos mix does not ───────
def test_image_tone_is_rejected_by_complex_mix():
    desired = _tone_out(45_000.0, 40_000.0)   # LO + 5 kHz -> 5 kHz
    image = _tone_out(35_000.0, 40_000.0)     # LO - 5 kHz -> must NOT appear
    f_d, db_d = _spectrum_db(desired, FS_OUT)
    f_i, db_i = _spectrum_db(image, FS_OUT)
    suppression = _band_peak_db(f_d, db_d, 5_000.0) - _band_peak_db(f_i, db_i, 5_000.0)
    assert suppression >= 50.0  # measured ~61 dB with the 255-tap one-sided FIR


def test_prohibited_cos_mix_shows_the_image_complex_mix_removes():
    # Same 35 kHz image tone through the real-cos reference: full-strength fold.
    cos_img = _cos_mix_reference(35_000.0, 40_000.0)
    cos_des = _cos_mix_reference(45_000.0, 40_000.0)
    f_ci, db_ci = _spectrum_db(cos_img, FS_OUT)
    f_cd, db_cd = _spectrum_db(cos_des, FS_OUT)
    # cos mix: image is as loud as the desired tone (within 3 dB) — the defect
    assert abs(_band_peak_db(f_ci, db_ci, 5_000.0) - _band_peak_db(f_cd, db_cd, 5_000.0)) < 3.0
    # complex mix kills the same input by >= 50 dB relative to the cos fold
    cplx_img = _tone_out(35_000.0, 40_000.0)
    f_xi, db_xi = _spectrum_db(cplx_img, FS_OUT)
    assert _band_peak_db(f_ci, db_ci, 5_000.0) - _band_peak_db(f_xi, db_xi, 5_000.0) >= 50.0


# ── continuity test (§9): block boundaries are seamless ─────────────────────
def test_chunked_processing_equals_single_call():
    sig = synth_signal(125_000, FS, kind="chirp", chirp_hz=(38_000.0, 52_000.0))
    whole = HeterodyneAudifier()
    whole.configure(40_000.0, 10_000.0, FS)
    out_whole = whole.process(sig)

    chunked = HeterodyneAudifier()
    chunked.configure(40_000.0, 10_000.0, FS)
    pieces, pos = [], 0
    for size in (1, 100, 2047, 256, 4096, 30_000, 88_500):  # uneven, sums to 125k
        pieces.append(chunked.process(sig[pos:pos + size]))
        pos += size
    assert pos == sig.size
    out_chunked = np.concatenate(pieces)

    # NCO phase, FIR zi, decimation phase, soxr state all carried -> identical
    assert out_chunked.size == out_whole.size
    assert np.allclose(out_chunked, out_whole, atol=1e-5)  # measured ~1.2e-7


# ── real capture (saved WAV with the ~25.43 kHz pest repeller) ──────────────
@pytest.mark.skipif(not WAV.exists(), reason="gitignored real-capture WAV not present")
def test_real_wav_repeller_maps_into_audible_band():
    from scipy.io import wavfile

    rate, data = wavfile.read(str(WAV))
    out = ddc_heterodyne(data.astype(np.float32), float(rate), 20_000.0, 10_000.0)
    freqs, db = _spectrum_db(out, FS_OUT)
    peak = freqs[int(np.argmax(db))]
    # M0 offline FFT measured the repeller at 25.43 kHz -> 25.43k - 20k = 5.43 kHz
    assert 5_200.0 < peak < 5_700.0
    power = 10.0 ** (db / 10.0)
    band = power[(freqs >= 3_000.0) & (freqs <= 11_000.0)].sum() / power.sum()
    assert band > 0.6  # measured ~78% of output energy in the mapped band


# ── one-shot flush (review): soxr tail must not be silently dropped ─────────
def test_oneshot_flush_recovers_resampler_tail():
    sig = synth_signal(int(FS), FS, tone_hz=45_000.0, amplitude=0.5)  # 1.0 s
    aud = HeterodyneAudifier()
    aud.configure(40_000.0, 10_000.0, FS)
    body = aud.process(sig)
    tail = aud.flush()
    assert tail.size > 300  # ~450 samples were held inside the streaming resampler
    # 250k in -> 50k decimated -> exactly 48000 out once flushed
    assert 47_995 <= body.size + tail.size <= 48_005
    assert ddc_heterodyne(sig, FS, 40_000.0, 10_000.0).size == body.size + tail.size


def test_flush_ends_stream_until_reconfigured():
    aud = HeterodyneAudifier()
    with pytest.raises(RuntimeError):
        aud.flush()  # before configure
    aud.configure(40_000.0, 10_000.0, FS)
    aud.process(synth_signal(10_000, FS, tone_hz=45_000.0))
    aud.flush()
    with pytest.raises(RuntimeError):
        aud.process(np.zeros(16, dtype=np.float32))  # soxr stream is finished
    aud.configure(40_000.0, 10_000.0, FS)  # reconfigure revives the instance
    assert aud.process(synth_signal(1_000, FS, tone_hz=45_000.0)).dtype == np.float32


# ── contract / validation ───────────────────────────────────────────────────
def test_conforms_to_audifier_protocol():
    assert isinstance(HeterodyneAudifier(), Audifier)


def test_process_before_configure_raises():
    with pytest.raises(RuntimeError):
        HeterodyneAudifier().process(np.zeros(16, dtype=np.float32))


def test_configure_validation():
    aud = HeterodyneAudifier()
    with pytest.raises(ValueError):
        aud.configure(40_000.0, 0.0, FS)          # zero bandwidth
    with pytest.raises(ValueError):
        aud.configure(-1.0, 10_000.0, FS)         # negative LO
    with pytest.raises(ValueError):
        aud.configure(130_000.0, 10_000.0, FS)    # LO beyond Nyquist
    with pytest.raises(ValueError):
        aud.configure(40_000.0, 10_000.0, 250_001.0)  # not divisible by DECIM


def test_configure_rejects_selection_wrapping_into_passband():
    # FATAL review finding: with f_lo + cutoff > fs/2 the mixed-down negative
    # line of a real tone wraps mod fs to fs - f - f_lo INSIDE the passband and
    # returns at full gain (the §11 defect through the back door).
    aud = HeterodyneAudifier()
    with pytest.raises(ValueError):
        aud.configure(120_000.0, 10_000.0, FS)   # 120k + 10k > 125k
    with pytest.raises(ValueError):
        aud.configure(110_000.0, 30_000.0, FS)   # cutoff clamps to 25k; 110k + 25k > 125k
    aud.configure(115_000.0, 10_000.0, FS)       # boundary f_lo + cutoff == fs/2 is legal


def test_configure_rejects_bandwidth_narrower_than_fir_transition():
    # Review finding: below one FIR transition width (~3.24 kHz at 250k/255 taps)
    # the "selection" would pass nothing but filter skirt.
    aud = HeterodyneAudifier()
    with pytest.raises(ValueError):
        aud.configure(40_000.0, 2_000.0, FS)
    aud.configure(40_000.0, 3_300.0, FS)         # just above the transition width


def test_failed_reconfigure_leaves_stream_intact():
    # configure() validates everything before mutating: a refused re-selection
    # must leave the running stream exactly as it was (atomic configure).
    aud = HeterodyneAudifier()
    aud.configure(40_000.0, 10_000.0, FS)
    ref = HeterodyneAudifier()
    ref.configure(40_000.0, 10_000.0, FS)
    sig = synth_signal(20_000, FS, tone_hz=45_000.0)
    aud.process(sig)
    ref.process(sig)
    with pytest.raises(ValueError):
        aud.configure(120_000.0, 10_000.0, FS)   # rejected mid-stream
    sig2 = synth_signal(20_000, FS, tone_hz=44_000.0)
    assert np.array_equal(aud.process(sig2), ref.process(sig2))


def test_empty_block_is_passthrough():
    aud = HeterodyneAudifier()
    aud.configure(40_000.0, 10_000.0, FS)
    out = aud.process(np.empty(0, dtype=np.float32))
    assert out.size == 0 and out.dtype == np.float32


def test_reconfigure_resets_stream_state():
    aud = HeterodyneAudifier()
    aud.configure(40_000.0, 10_000.0, FS)
    aud.process(synth_signal(10_000, FS, tone_hz=45_000.0))
    aud.configure(20_000.0, 10_000.0, FS)  # re-select band mid-stream
    fresh = HeterodyneAudifier()
    fresh.configure(20_000.0, 10_000.0, FS)
    sig = synth_signal(50_000, FS, tone_hz=25_000.0)
    assert np.array_equal(aud.process(sig), fresh.process(sig))
