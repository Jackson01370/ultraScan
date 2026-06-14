"""M3 GainStage / AGCGain — small-signal amplification (DESIGN §4.3 / §6 M3).

Numeric, Sim-only (the milestone has a light DSP contract, so verification is
numeric rather than a long review): AGC lifts a quiet signal toward the target,
attenuates a loud one, keeps the gain CONTINUOUS across block boundaries (no
click), attacks faster than it releases, and — by design — lifts the noise floor
with the signal (the documented physical limit; AGC is not a noise gate).
"""

import time

import numpy as np

from ultrascan.audio.spsc import SpscAudioRing
from ultrascan.audio.worker import AudioWorker
from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.capture.sources import synth_signal
from ultrascan.dsp.audifier import HeterodyneAudifier
from ultrascan.dsp.gain import AGCGain, GainStage

FS = 48_000.0
FS_IN = 250_000.0


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# ── protocol / construction ──────────────────────────────────────────────────
def test_agc_satisfies_gainstage_protocol():
    agc = AGCGain(FS)
    assert isinstance(agc, GainStage)


def test_constructor_rejects_bad_params():
    for kwargs in (
        dict(fs=0.0),
        dict(fs=FS, target_rms=0.0),
        dict(fs=FS, attack_s=-1.0),
        dict(fs=FS, max_gain=0.0),
    ):
        try:
            AGCGain(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs!r}")


def test_empty_block_passes_through():
    agc = AGCGain(FS)
    out = agc.process(np.empty(0, dtype=np.float32))
    assert out.size == 0
    assert agc.current_gain == 1.0  # untouched


# ── core behaviour: lift quiet, attenuate loud ───────────────────────────────
def test_quiet_signal_is_lifted_toward_target():
    """A faint tone: OFF (identity) stays faint; ON climbs toward target_rms."""
    n = int(0.5 * FS)
    quiet = (0.01 * np.sin(2 * np.pi * 5_000.0 * np.arange(n) / FS)).astype(np.float32)
    in_rms = _rms(quiet)

    agc = AGCGain(FS, target_rms=0.2, release_s=0.05)
    out = np.concatenate([agc.process(quiet[i:i + 1024]) for i in range(0, n, 1024)])

    assert in_rms < 0.02                       # genuinely faint
    assert _rms(out) > 8 * in_rms              # ON lifts it a lot
    # converges near the target by the tail (within a factor ~1.5)
    tail = out[-int(0.1 * FS):]
    assert 0.13 < _rms(tail) < 0.30
    assert agc.current_gain > 5.0


def test_loud_signal_is_attenuated_toward_target():
    n = int(0.5 * FS)
    loud = (0.9 * np.sin(2 * np.pi * 5_000.0 * np.arange(n) / FS)).astype(np.float32)
    agc = AGCGain(FS, target_rms=0.2)
    out = np.concatenate([agc.process(loud[i:i + 1024]) for i in range(0, n, 1024)])
    tail = out[-int(0.1 * FS):]
    assert _rms(tail) < _rms(loud[-int(0.1 * FS):])  # pulled DOWN
    assert 0.13 < _rms(tail) < 0.30                  # near target
    assert agc.current_gain < 1.0


def test_steady_signal_converges_to_target():
    n = int(2.0 * FS)
    sig = (0.05 * np.sin(2 * np.pi * 4_000.0 * np.arange(n) / FS)).astype(np.float32)
    agc = AGCGain(FS, target_rms=0.2, release_s=0.1)
    out = np.concatenate([agc.process(sig[i:i + 2048]) for i in range(0, n, 2048)])
    assert abs(_rms(out[-int(0.2 * FS):]) - 0.2) < 0.03


# ── continuity: gain is continuous across block boundaries (no click) ─────────
def test_gain_is_continuous_across_block_boundaries():
    """Reconstruct the per-sample gain from a constant input (out/in) and assert
    it has no jump at chunk boundaries — the M3 continuity requirement."""
    n = 40_000
    x = np.full(n, 0.01, dtype=np.float32)  # constant -> out/in IS the gain
    agc = AGCGain(FS, target_rms=0.2, release_s=0.2)

    chunks, sizes, bounds = [], [512, 1024, 2048, 256, 4096], []
    i = 0
    k = 0
    while i < n:
        m = min(sizes[k % len(sizes)], n - i)
        chunks.append(agc.process(x[i:i + m]))
        i += m
        bounds.append(i)  # absolute index of each block's last sample (+1)
        k += 1
    out = np.concatenate(chunks)
    gain = out / 0.01  # per-sample applied gain

    steps = np.abs(np.diff(gain))
    # The gain ramps linearly inside each block AND g_new(j) == g_start(j+1) at
    # every boundary => the whole curve is continuous. A click (state not carried)
    # would show up as a boundary step of order the gain magnitude (~unity+).
    # Assert each boundary step is no bigger than the surrounding ramp slope.
    for b in bounds[:-1]:
        boundary_step = steps[b - 1]
        local = steps[max(0, b - 5):b + 5].max()
        assert boundary_step <= local + 1e-6, (
            f"boundary discontinuity at {b}: {boundary_step:.2e} >> local {local:.2e}"
        )
    assert steps.max() < 0.01, f"gain step too large {steps.max():.2e}"
    assert gain[0] >= 1.0 and gain[-1] > gain[0]  # rose (release) toward target


def test_state_persists_across_blocks():
    """g at the start of block N+1 equals g at the end of block N."""
    x = np.full(2000, 0.02, dtype=np.float32)
    agc = AGCGain(FS, target_rms=0.2, release_s=0.2)
    out1 = agc.process(x[:1000])
    g_after_1 = agc.current_gain
    assert abs(out1[-1] / 0.02 - g_after_1) < 1e-4   # block 1 ends AT g_after_1
    out2 = agc.process(x[1000:])
    assert abs(out2[0] / 0.02 - g_after_1) < 1e-3    # block 2 STARTS at g_after_1


# ── attack ≪ release (overshoot vs pumping) ──────────────────────────────────
def test_attack_is_faster_than_release():
    """One block of equal length: the fractional move toward the wanted gain is
    larger when REDUCING gain (attack) than when RAISING it (release)."""
    n = 480  # 10 ms @ 48k

    # attack: loud signal -> wanted gain < 1, current gain starts at 1.0
    a = AGCGain(FS, target_rms=0.2, attack_s=0.010, release_s=0.300)
    loud = np.full(n, 0.9, dtype=np.float32)
    a.process(loud)
    g_want_attack = 0.2 / 0.9
    frac_attack = (1.0 - a.current_gain) / (1.0 - g_want_attack)

    # release: quiet signal -> wanted gain > 1, current gain starts at 1.0
    r = AGCGain(FS, target_rms=0.2, attack_s=0.010, release_s=0.300)
    quiet = np.full(n, 0.02, dtype=np.float32)
    r.process(quiet)
    g_want_release = min(0.2 / 0.02, 100.0)
    frac_release = (r.current_gain - 1.0) / (g_want_release - 1.0)

    assert frac_attack > frac_release          # attack moves more in the same time
    assert frac_attack > 0.5                    # 10 ms block ≈ one attack tau -> ~63%


# ── documented physical limit: gain ceiling + noise floor rises ──────────────
def test_silence_gain_is_capped_at_max_gain():
    agc = AGCGain(FS, target_rms=0.2, release_s=0.05, max_gain=50.0)
    silence = np.zeros(2048, dtype=np.float32)
    for _ in range(200):
        agc.process(silence)
    assert agc.current_gain <= 50.0 + 1e-6
    assert agc.current_gain > 40.0  # climbed toward the ceiling, did not blow up


def test_noise_floor_rises_with_gain():
    """DESIGN §6 M3 constraint, made explicit: AGC lifts a faint noise floor
    toward the target just like a faint signal — it is NOT a noise gate."""
    n = int(0.5 * FS)
    # deterministic faint "noise" (no global RNG): sum of incommensurate tones
    t = np.arange(n) / FS
    noise = 0.005 * (np.sin(2 * np.pi * 3_111 * t) + np.sin(2 * np.pi * 7_333 * t)
                     + np.sin(2 * np.pi * 11_777 * t)).astype(np.float32)
    in_rms = _rms(noise)
    agc = AGCGain(FS, target_rms=0.2, release_s=0.05)
    out = np.concatenate([agc.process(noise[i:i + 1024]) for i in range(0, n, 1024)])
    assert _rms(out[-int(0.1 * FS):]) > 10 * in_rms  # the floor came up with it


def test_reset_returns_gain_to_unity():
    agc = AGCGain(FS, target_rms=0.2)
    agc.process(np.full(4096, 0.9, dtype=np.float32))
    assert agc.current_gain != 1.0
    agc.reset()
    assert agc.current_gain == 1.0


# ── worker integration: DDC -> AGC -> SPSC lifts a faint band ─────────────────
def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _drain_all(ring):
    chunks = []
    while True:
        buf = np.empty(4096, dtype=np.float32)
        real = ring.pop_into(buf)
        if real == 0:
            break
        chunks.append(buf[:real].copy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _run_worker(gain):
    n = int(1.0 * FS_IN)
    # faint 45 kHz tone (amplitude 0.01) -> DDC maps to ~5 kHz, quiet
    sig = synth_signal(n, FS_IN, tone_hz=45_000.0, amplitude=0.01)
    l1 = RingBuffer(n)
    l1.write(sig)
    out_ring = SpscAudioRing(capacity=128_000, prebuffer=0)
    worker = AudioWorker(
        l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0, gain=gain,
    )
    worker.start()
    assert _wait_until(lambda: worker.n_in_samples >= n)
    worker.stop()
    return _drain_all(out_ring)


def test_worker_with_agc_lifts_faint_band():
    plain = _run_worker(gain=None)
    boosted = _run_worker(gain=AGCGain(FS, target_rms=0.2, release_s=0.1))
    assert plain.size > 40_000 and boosted.size > 40_000
    assert _rms(plain) < 0.02                       # faint without AGC
    assert _rms(boosted) > 6 * _rms(plain)          # AGC lifted it
    # peak still at 5 kHz: AGC is a level stage, it does not move frequencies
    spec = np.abs(np.fft.rfft(boosted * np.hanning(boosted.size)))
    peak_hz = np.argmax(spec) * FS / boosted.size
    assert abs(peak_hz - 5_000.0) < 50.0
