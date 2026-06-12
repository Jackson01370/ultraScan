"""AudioWorker — L1 cursor -> frozen HeterodyneAudifier -> SPSC, plus band swaps.

DDC numerical correctness itself is pinned by the M2a tests; here we pin the
*plumbing*: the worker-driven streaming output must equal a one-shot reference
run of the same frozen audifier over the same input, and band re-selection /
refusal / input-gap behaviour must follow M2b click policy (a).
"""

import time

import numpy as np

from ultrascan.audio.spsc import SpscAudioRing
from ultrascan.audio.worker import AudioWorker
from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.capture.sources import synth_signal
from ultrascan.dsp.audifier import HeterodyneAudifier

FS_IN = 250_000.0


def _wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _drain_all(ring: SpscAudioRing) -> np.ndarray:
    chunks = []
    while True:
        buf = np.empty(4096, dtype=np.float32)
        real = ring.pop_into(buf)
        if real == 0:
            break
        chunks.append(buf[:real].copy())
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def test_worker_stream_equals_oneshot_reference():
    n = int(1.0 * FS_IN)
    sig = synth_signal(n, FS_IN, tone_hz=45_000.0)
    l1 = RingBuffer(n)
    l1.write(sig)

    out_ring = SpscAudioRing(capacity=64_000, prebuffer=0)
    worker = AudioWorker(
        l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0,
    )
    worker.start()
    assert _wait_until(lambda: worker.n_in_samples >= n)
    worker.stop()
    got = _drain_all(out_ring)

    ref_aud = HeterodyneAudifier()
    ref_aud.configure(40_000.0, 10_000.0, FS_IN)
    ref = ref_aud.process(sig)
    # Same frozen DDC, different chunking — M2a chunking-invariance bounds the diff.
    m = min(got.size, ref.size)
    assert m > 40_000
    np.testing.assert_allclose(got[:m], ref[:m], atol=1e-5)

    # 45 kHz tone with LO 40 kHz must land at 5 kHz audible.
    spec = np.abs(np.fft.rfft(got * np.hanning(got.size)))
    peak_hz = np.argmax(spec) * 48_000.0 / got.size
    assert abs(peak_hz - 5_000.0) < 50.0


def test_band_change_applies_between_blocks_and_keeps_streaming():
    n_half = int(0.3 * FS_IN)
    sig = synth_signal(2 * n_half, FS_IN, tone_hz=45_000.0)
    l1 = RingBuffer(2 * n_half)
    out_ring = SpscAudioRing(capacity=64_000, prebuffer=0)
    worker = AudioWorker(
        l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0,
    )
    worker.start()
    l1.write(sig[:n_half])
    assert _wait_until(lambda: worker.n_in_samples >= n_half)
    out_before = worker.n_out_samples

    worker.request_band(38_000.0, 10_000.0)  # valid: 45k tone -> 7 kHz
    l1.write(sig[n_half:])
    assert _wait_until(lambda: worker.n_in_samples >= 2 * n_half)
    assert _wait_until(lambda: worker.n_band_changes == 1)
    assert worker.band == (38_000.0, 10_000.0)
    assert worker.last_band_error is None
    assert _wait_until(lambda: worker.n_out_samples > out_before)  # kept streaming
    worker.stop()


def test_refused_band_keeps_old_band_and_stream():
    n = int(0.3 * FS_IN)
    sig = synth_signal(n, FS_IN, tone_hz=45_000.0)
    l1 = RingBuffer(2 * n)
    out_ring = SpscAudioRing(capacity=64_000, prebuffer=0)
    worker = AudioWorker(
        l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0,
    )
    worker.start()
    l1.write(sig)
    assert _wait_until(lambda: worker.n_in_samples >= n)

    # f_lo + cutoff > fs/2: the M2a fatal-finding guard must refuse this.
    worker.request_band(120_000.0, 10_000.0)
    assert _wait_until(lambda: worker.last_band_error is not None)
    assert "Nyquist" in worker.last_band_error
    assert worker.band == (40_000.0, 10_000.0)  # old band still configured
    assert worker.n_band_changes == 0

    out_before = worker.n_out_samples
    l1.write(sig)
    assert _wait_until(lambda: worker.n_out_samples > out_before)  # still alive
    worker.stop()


def test_lapped_reader_gap_is_counted_and_stream_survives():
    # L1 ring far smaller than the backlog -> the audio reader gets lapped.
    l1 = RingBuffer(8_192)
    out_ring = SpscAudioRing(capacity=256_000, prebuffer=0)
    worker = AudioWorker(
        l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0,
        poll_s=0.05,  # slow poll so the writer below laps it deterministically
    )
    sig = synth_signal(int(0.5 * FS_IN), FS_IN, tone_hz=45_000.0)
    worker.start()
    for off in range(0, sig.size, 4_096):
        l1.write(sig[off:off + 4_096])  # much faster than the worker polls
    assert _wait_until(lambda: worker.n_dropped_in > 0)
    assert _wait_until(lambda: worker.n_out_samples > 0)  # gap -> click, not death
    worker.stop()


def test_volume_is_plain_attenuation():
    n = int(0.2 * FS_IN)
    sig = synth_signal(n, FS_IN, tone_hz=45_000.0)

    def run(volume):
        l1 = RingBuffer(n)
        l1.write(sig)
        out_ring = SpscAudioRing(capacity=64_000, prebuffer=0)
        worker = AudioWorker(
            l1.reader(from_now=False), HeterodyneAudifier(), out_ring, FS_IN,
            f_lo_sel=40_000.0, bandwidth=10_000.0, volume=volume,
        )
        worker.start()
        assert _wait_until(lambda: worker.n_in_samples >= n)
        worker.stop()
        return _drain_all(out_ring)

    full = run(1.0)
    half = run(0.5)
    m = min(full.size, half.size)
    np.testing.assert_allclose(half[:m], 0.5 * full[:m], atol=1e-6)
