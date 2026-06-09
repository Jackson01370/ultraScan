"""Sim-first tests for the M0 stress-test core (no hardware required)."""

import numpy as np

from ultrascan.capture.sources import SyntheticSource
from ultrascan.capture.stress import render_report, run_capture


def _synth(**kw):
    return SyntheticSource(samplerate=250_000.0, blocksize=2048, tone_hz=45_000.0, **kw)


def test_run_capture_detects_ultrasonic_energy():
    result = run_capture(_synth(), duration=0.4, load_ms=0.0)
    assert result.total_frames > 0
    assert result.n_callbacks > 5
    # 45 kHz tone -> peak near 45 kHz and energy is overwhelmingly above 24 kHz.
    assert abs(result.peak_hz - 45_000.0) < 200.0
    assert result.energy_above_guard_frac > 0.9
    assert result.has_ultrasonic_energy
    # cheap consumer, no injected load -> no simulated overruns
    assert result.overflow_count == 0


def test_dummy_load_beyond_block_period_causes_overruns():
    # block period @ 2048/250k ~= 8.19 ms; a 20 ms load must overrun every block.
    result = run_capture(_synth(), duration=0.3, load_ms=20.0)
    assert result.overflow_count > 0
    assert result.service_ms_max >= 20.0


def test_report_renders_synthetic_banner(tmp_path):
    result = run_capture(_synth(), duration=0.25, load_ms=0.0)
    md = render_report(result)
    assert "SYNTHETIC-ONLY" in md
    assert "FFT verification" in md
    assert "input_overflow (Xrun)" in md
    # report is writable as UTF-8
    out = tmp_path / "report.md"
    out.write_text(md, encoding="utf-8")
    assert out.read_text(encoding="utf-8") == md


def test_no_ultrasonic_energy_for_low_tone():
    # A 5 kHz tone has no energy above the 24 kHz guard -> verdict NO.
    src = SyntheticSource(samplerate=250_000.0, blocksize=2048, tone_hz=5_000.0)
    result = run_capture(src, duration=0.3, load_ms=0.0)
    assert abs(result.peak_hz - 5_000.0) < 200.0
    assert result.energy_above_guard_frac < 0.01
    assert not result.has_ultrasonic_energy
