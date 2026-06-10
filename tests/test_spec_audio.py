"""M1 tests: the FROZEN spec_audio.render contract (DESIGN §4.1).

These pin the frozen behaviour: [256,256] float32 [0,1], rfft one-sided,
DB_DYN_RANGE=60, rows = frequency (row 0 = DC), columns = time.
"""

import numpy as np
import pytest

from ultrascan.capture.sources import synth_signal
from ultrascan.dsp.spec_audio import DB_DYN_RANGE, NFFT, OUT_SHAPE, render

FS = 250_000.0


def test_contract_shape_dtype_range():
    sig = synth_signal(50_000, FS, tone_hz=45_000.0)
    out = render(sig, FS)
    assert out.shape == OUT_SHAPE == (256, 256)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0
    assert float(out.max()) == pytest.approx(1.0)  # loudest bin is the 0 dB reference


def test_tone_lands_on_expected_frequency_row():
    tone = 45_000.0
    sig = synth_signal(100_000, FS, tone_hz=tone)
    out = render(sig, FS)
    expected_row = int(round(tone / FS * NFFT))  # bin = f / rate * NFFT
    peak_rows = np.argmax(out, axis=0)  # per time column
    assert np.all(np.abs(peak_rows - expected_row) <= 1)


def test_chirp_ridge_moves_across_time():
    sig = synth_signal(250_000, FS, kind="chirp", chirp_hz=(20_000.0, 90_000.0))
    out = render(sig, FS)
    peak_rows = np.argmax(out, axis=0).astype(float)
    # 20->90 kHz sweep: the ridge must rise substantially from early to late columns.
    assert peak_rows[224:].mean() - peak_rows[:32].mean() > 50


def test_all_zero_input_renders_all_zero():
    out = render(np.zeros(10_000, dtype=np.float32), FS)
    assert out.shape == OUT_SHAPE
    assert not out.any()


def test_deterministic_bitwise():
    sig = synth_signal(30_000, FS, kind="chirp")
    a = render(sig, FS)
    b = render(sig.copy(), FS)
    assert np.array_equal(a, b)


def test_short_input_is_padded_to_one_frame():
    sig = synth_signal(NFFT // 4, FS, tone_hz=40_000.0)  # shorter than one window
    out = render(sig, FS)
    assert out.shape == OUT_SHAPE
    assert float(out.max()) == pytest.approx(1.0)


def test_dynamic_range_floor_maps_to_zero():
    # Tone + silence-dominated frames: floor values clip exactly to 0 after the
    # -DB_DYN_RANGE clamp, so the output uses the full [0,1] interval.
    sig = np.concatenate([
        synth_signal(20_000, FS, tone_hz=45_000.0),
        np.zeros(80_000, dtype=np.float32),
    ])
    out = render(sig, FS)
    assert float(out.min()) == 0.0
    assert float(out.max()) == pytest.approx(1.0)
    assert DB_DYN_RANGE == 60.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        render(np.empty(0, dtype=np.float32), FS)
    with pytest.raises(ValueError):
        render(synth_signal(1000, FS), 0.0)
