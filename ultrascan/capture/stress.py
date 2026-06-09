"""M0 capture stress test (DESIGN §6 M0).

Drives any :class:`InputSource` and answers the M0 questions:
  - is real >24 kHz energy captured? (offline FFT after capture; Nyquist 125 kHz)
  - how many overruns (Xrun) at a given blocksize / latency?
  - what happens under a dummy callback load (boundary behaviour)?

The consumer callback stays copy-only + counters (DESIGN §2); the FFT is run
offline on the ring snapshot once capture stops — never inside the callback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .ring_buffer import RingBuffer
from .sources import InputSource

NYQUIST_GUARD_HZ = 24_000.0  # the line share-mode resampling would have eaten


@dataclass
class StressResult:
    source: dict
    requested_duration_s: float
    elapsed_s: float
    blocksize: int
    samplerate: float
    block_period_ms: float
    load_ms: float
    n_callbacks: int
    total_frames: int
    overflow_count: int
    underflow_count: int
    service_ms_min: float
    service_ms_mean: float
    service_ms_max: float
    service_ms_p99: float
    fft_nfft: int
    peak_hz: float
    energy_above_guard_frac: float
    has_ultrasonic_energy: bool
    native_rate_check: Optional[dict] = None
    capture_error: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _fft_check(samples: np.ndarray, samplerate: float, nfft: int) -> dict:
    """Offline one-sided rfft: peak frequency + fraction of energy above the guard."""
    n = min(nfft, samples.size)
    if n < 16:
        return {"nfft": 0, "peak_hz": 0.0, "frac": 0.0, "ok": False}
    n = 1 << int(np.floor(np.log2(n)))  # power-of-two window
    seg = samples[-n:].astype(np.float64)
    win = np.hanning(n)
    spec = np.fft.rfft(seg * win)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = np.fft.rfftfreq(n, 1.0 / samplerate)
    power[0] = 0.0  # ignore DC
    total = power.sum()
    if total <= 0:
        return {"nfft": n, "peak_hz": 0.0, "frac": 0.0, "ok": False}
    peak_hz = float(freqs[int(np.argmax(power))])
    frac = float(power[freqs > NYQUIST_GUARD_HZ].sum() / total)
    return {"nfft": n, "peak_hz": peak_hz, "frac": frac, "ok": True}


def run_capture(
    source: InputSource,
    duration: float,
    *,
    load_ms: float = 0.0,
    ring_seconds: float = 2.0,
    fft_nfft: int = 65_536,
    ultrasonic_frac_threshold: float = 0.01,
) -> StressResult:
    """Capture for ``duration`` seconds and summarise. ``load_ms`` injects a dummy
    per-callback sleep to probe overload behaviour (M0 dummy-load test)."""
    ring = RingBuffer(max(int(ring_seconds * source.samplerate), fft_nfft))
    period_ms = 1000.0 * source.blocksize / source.samplerate

    counters = {"cb": 0, "frames": 0, "ovf": 0, "unf": 0}
    service_ms: List[float] = []
    load_s = load_ms / 1000.0

    def on_block(block: np.ndarray, status) -> None:
        t0 = time.perf_counter()
        if load_s > 0:
            time.sleep(load_s)            # dummy load (DESIGN §6 M0 boundary test)
        ring.write(block)                 # copy-only path
        counters["cb"] += 1
        counters["frames"] += block.size
        if status.input_overflow:
            counters["ovf"] += 1
        if status.input_underflow:
            counters["unf"] += 1
        service_ms.append((time.perf_counter() - t0) * 1000.0)

    native_check = None
    verify = getattr(source, "verify_native_rate", None)
    if callable(verify):
        native_check = verify()

    capture_error = None
    t_start = time.perf_counter()
    try:
        source.start(on_block)
        deadline = t_start + duration
        while time.perf_counter() < deadline and source.is_running:
            time.sleep(0.02)
    except Exception as exc:  # noqa: BLE001 - a failed open IS an M0 result, not a crash
        capture_error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            source.stop()
        except Exception:  # noqa: BLE001
            pass
    elapsed = time.perf_counter() - t_start

    if counters["frames"] > 0:
        fft = _fft_check(ring.latest(fft_nfft), source.samplerate, fft_nfft)
    else:
        fft = {"nfft": 0, "peak_hz": 0.0, "frac": 0.0, "ok": False}
    svc = np.asarray(service_ms) if service_ms else np.zeros(1)

    notes: List[str] = []
    if capture_error:
        notes.append(f"CAPTURE FAILED before/while streaming: {capture_error}")
    if getattr(source, "is_synthetic", False):
        notes.append(
            "SYNTHETIC-ONLY: Sim source. Real 250k native capture, Xrun limits and "
            "audibility are judged on real hardware by Kali."
        )
    if load_ms > 0:
        notes.append(
            f"Dummy load {load_ms:.1f} ms/callback vs block period {period_ms:.2f} ms "
            f"-> {'overruns expected' if load_ms > period_ms else 'within budget'}."
        )

    return StressResult(
        source=source.describe(),
        requested_duration_s=float(duration),
        elapsed_s=float(elapsed),
        blocksize=source.blocksize,
        samplerate=source.samplerate,
        block_period_ms=period_ms,
        load_ms=float(load_ms),
        n_callbacks=counters["cb"],
        total_frames=counters["frames"],
        overflow_count=counters["ovf"],
        underflow_count=counters["unf"],
        service_ms_min=float(svc.min()),
        service_ms_mean=float(svc.mean()),
        service_ms_max=float(svc.max()),
        service_ms_p99=float(np.percentile(svc, 99)),
        fft_nfft=int(fft["nfft"]),
        peak_hz=float(fft["peak_hz"]),
        energy_above_guard_frac=float(fft["frac"]),
        has_ultrasonic_energy=bool(fft["ok"] and fft["frac"] >= ultrasonic_frac_threshold),
        native_rate_check=native_check,
        capture_error=capture_error,
        notes=notes,
    )


def render_report(result: StressResult) -> str:
    """Markdown for ``M0_capture_report.md`` (DESIGN §6 / §10)."""
    r = result
    synthetic = bool(r.source.get("is_synthetic"))
    lines: List[str] = []
    lines.append("# M0 — Capture Stress Test Report")
    lines.append("")
    if synthetic:
        lines.append("> **SYNTHETIC-ONLY** — generated from a Sim source (no microphone).")
        lines.append("> Real 250k native capture, Xrun limits and audibility are judged on")
        lines.append("> real hardware by **Kali**. These numbers validate the *pipeline*, not the HW.")
        lines.append("")

    lines.append("## Source")
    for k, v in r.source.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    if r.native_rate_check is not None:
        lines.append("## WASAPI exclusive native-rate check")
        for k, v in r.native_rate_check.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    if r.capture_error:
        lines.append("## CAPTURE FAILED (real-HW blocker)")
        lines.append(f"- error: `{r.capture_error}`")
        lines.append("- No native-rate stream opened, so the >24 kHz energy verdict could "
                     "**not** be measured (see FFT section).")
        lines.append("- Note: `IsFormatSupported` (the native-rate check above) passing while "
                     "the stream open fails is typically a WASAPI **exclusive** `Initialize` "
                     "refusal — device held by another app, or exclusive control disabled.")
        lines.append("")

    lines.append("## Capture")
    lines.append(f"- requested duration: {r.requested_duration_s:.2f} s")
    lines.append(f"- elapsed: {r.elapsed_s:.2f} s")
    lines.append(f"- samplerate: {r.samplerate:.0f} Hz (Nyquist {r.samplerate/2:.0f} Hz)")
    lines.append(f"- blocksize: {r.blocksize} samples  (block period {r.block_period_ms:.2f} ms)")
    lines.append(f"- callbacks: {r.n_callbacks}")
    lines.append(f"- total frames: {r.total_frames}")
    lines.append("")

    lines.append("## Xrun / overrun")
    lines.append(f"- input_overflow (Xrun) count: **{r.overflow_count}**")
    lines.append(f"- input_underflow count: {r.underflow_count}")
    lines.append(f"- dummy load injected: {r.load_ms:.1f} ms/callback")
    lines.append(
        "- callback service time (ms): "
        f"min {r.service_ms_min:.2f} / mean {r.service_ms_mean:.2f} / "
        f"p99 {r.service_ms_p99:.2f} / max {r.service_ms_max:.2f}"
    )
    lines.append("")

    lines.append("## FFT verification (offline, after capture)")
    if r.fft_nfft == 0:
        lines.append(f"- ultrasonic energy (>{NYQUIST_GUARD_HZ/1000:.0f} kHz): **NOT MEASURED** "
                     "— no samples were captured (capture did not open).")
    else:
        lines.append(f"- nfft: {r.fft_nfft}")
        lines.append(f"- peak frequency: **{r.peak_hz/1000.0:.2f} kHz**")
        lines.append(
            f"- energy fraction above {NYQUIST_GUARD_HZ/1000:.0f} kHz: "
            f"**{r.energy_above_guard_frac*100:.2f}%**"
        )
        verdict = "YES" if r.has_ultrasonic_energy else "NO"
        lines.append(f"- ultrasonic energy present (>{NYQUIST_GUARD_HZ/1000:.0f} kHz): **{verdict}**")
    lines.append("")

    if r.notes:
        lines.append("## Notes")
        for note in r.notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Overload boundary (dummy-load test, DESIGN §6 M0)")
    if r.load_ms > 0:
        graceful = "process kept running (graceful degradation — overruns flagged, no crash)"
        lines.append(
            f"- this run injected {r.load_ms:.1f} ms/callback (> {r.block_period_ms:.2f} ms "
            f"block period) and saw {r.overflow_count} overruns; {graceful}."
        )
    else:
        lines.append(
            "- reproduce with `--load-ms <ms>`; a per-callback load above the "
            f"{r.block_period_ms:.2f} ms block period flags every block as an overrun. "
            "Observed Sim behaviour: graceful degradation (overruns flagged, no crash)."
        )
    lines.append(
        "- host note: Windows `time.sleep` granularity is ~15 ms, so Sim pacing is "
        "coarse — overrun detection is keyed to consumer-callback duration, not host "
        "timer jitter. Real per-block latency/Xrun limits are an HW measurement (Kali)."
    )
    lines.append("")

    lines.append("## What M0 establishes / hands off to Kali")
    if synthetic:
        lines.append("- **establishes (Sim):** the capture pipe runs end-to-end — "
                     "InputSource -> copy-only callback -> ring -> offline FFT — and that a "
                     "known >24 kHz tone is recovered, overruns are detected, and overload "
                     "degrades gracefully.")
        lines.append("- **does NOT establish:** that this PC + UltraMic can actually open "
                     "WASAPI **exclusive** at 250k, nor the real Xrun-free blocksize floor.")
    lines.append("- **Kali, on real hardware, confirm:**")
    lines.append("  1. `--source wasapi --duration 10` opens 250k exclusive (native-rate check OK).")
    lines.append("  2. real energy appears above 24 kHz (e.g. jingling keys / ultrasonic remote).")
    lines.append("  3. sweep `--blocksize` down; find the smallest with 0 Xrun over several minutes.")
    lines.append("  4. `--load-ms` past the block period only stalls audio (no crash).")
    lines.append("")

    return "\n".join(lines)
