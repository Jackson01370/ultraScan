"""SpscAudioRing — FIFO exactness, priming, underrun accounting, backpressure."""

import threading
import time

import numpy as np
import pytest

from ultrascan.audio.spsc import SpscAudioRing


def _pop(ring, n):
    buf = np.empty(n, dtype=np.float32)
    real = ring.pop_into(buf)
    return buf, real


def test_priming_emits_silence_until_prebuffer():
    ring = SpscAudioRing(capacity=64, prebuffer=8)
    out, real = _pop(ring, 4)
    assert real == 0 and np.all(out == 0.0)
    assert ring.n_underruns == 0  # start-of-stream priming is not an underrun
    ring.push(np.arange(1, 7, dtype=np.float32))  # 6 < prebuffer: still priming
    out, real = _pop(ring, 4)
    assert real == 0 and np.all(out == 0.0)
    ring.push(np.array([7.0, 8.0], dtype=np.float32))  # occupancy hits 8
    out, real = _pop(ring, 4)
    assert real == 4
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0])
    assert ring.n_underruns == 0


def test_fifo_exact_across_wraparound():
    ring = SpscAudioRing(capacity=16, prebuffer=0)
    rng = np.random.default_rng(42)
    seq = np.arange(1, 501, dtype=np.float32)
    pos = 0
    collected = []
    while len(collected) < seq.size:
        free = ring.capacity - ring.occupancy
        n = int(rng.integers(1, 9))
        if pos < seq.size and rng.random() < 0.6 and free >= n:
            chunk = seq[pos:pos + n]
            assert ring.push(chunk)
            pos += chunk.size
        else:
            out, real = _pop(ring, n)
            collected.extend(out[:real].tolist())
    np.testing.assert_array_equal(np.array(collected, dtype=np.float32), seq)
    assert ring.n_pushed == seq.size
    assert ring.n_popped_real == seq.size


def test_underrun_zero_fills_counts_and_reprimes():
    ring = SpscAudioRing(capacity=64, prebuffer=4)
    ring.push(np.arange(1, 7, dtype=np.float32))  # 6 >= prebuffer
    out, real = _pop(ring, 10)  # primed but only 6 available -> underrun
    assert real == 6
    np.testing.assert_array_equal(out[:6], [1, 2, 3, 4, 5, 6])
    assert np.all(out[6:] == 0.0)
    assert ring.n_underruns == 1
    assert not ring.is_primed  # dropped back to priming
    ring.push(np.array([7.0, 8.0], dtype=np.float32))  # 2 < prebuffer
    out, real = _pop(ring, 4)
    assert real == 0 and np.all(out == 0.0)  # re-priming silence, not an underrun
    assert ring.n_underruns == 1
    ring.push(np.array([9.0, 10.0], dtype=np.float32))  # occupancy 4 -> primed
    out, real = _pop(ring, 4)
    assert real == 4
    np.testing.assert_array_equal(out, [7, 8, 9, 10])


def test_push_backpressure_blocks_until_consumer_frees_space():
    ring = SpscAudioRing(capacity=32, prebuffer=0)
    big = np.arange(1, 101, dtype=np.float32)  # 100 > capacity: needs draining
    done = {"ok": None}

    def producer():
        done["ok"] = ring.push(big, timeout_s=5.0)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    collected = []
    deadline = time.monotonic() + 5.0
    while len(collected) < big.size and time.monotonic() < deadline:
        out, real = _pop(ring, 16)
        collected.extend(out[:real].tolist())
        time.sleep(0.001)
    t.join(timeout=5.0)
    assert done["ok"] is True
    np.testing.assert_array_equal(np.array(collected, dtype=np.float32), big)


def test_push_timeout_returns_false_when_full():
    ring = SpscAudioRing(capacity=8, prebuffer=0)
    assert ring.push(np.ones(8, dtype=np.float32))  # fills the ring
    t0 = time.monotonic()
    ok = ring.push(np.ones(4, dtype=np.float32), timeout_s=0.1)
    assert ok is False
    assert time.monotonic() - t0 < 2.0


def test_push_stop_event_aborts():
    ring = SpscAudioRing(capacity=8, prebuffer=0)
    ring.push(np.ones(8, dtype=np.float32))
    stop = threading.Event()
    stop.set()
    assert ring.push(np.ones(4, dtype=np.float32), stop_event=stop) is False


def test_pop_larger_than_capacity_is_clamped_and_zero_filled():
    ring = SpscAudioRing(capacity=8, prebuffer=0)
    ring.push(np.arange(1, 9, dtype=np.float32))
    out, real = _pop(ring, 20)
    assert real == 8
    np.testing.assert_array_equal(out[:8], np.arange(1, 9, dtype=np.float32))
    assert np.all(out[8:] == 0.0)


def test_validation():
    with pytest.raises(ValueError):
        SpscAudioRing(capacity=0, prebuffer=0)
    with pytest.raises(ValueError):
        SpscAudioRing(capacity=8, prebuffer=9)


def test_threaded_producer_consumer_preserves_order():
    ring = SpscAudioRing(capacity=256, prebuffer=32)
    n_total = 20_000
    seq = np.arange(1, n_total + 1, dtype=np.float32)  # no zeros: zero == filler
    stop = threading.Event()

    def producer():
        # Production semantics (AudioWorker): no timeout, stop_event only — a
        # timed-out partial push retried would duplicate samples, so don't.
        pos = 0
        while pos < n_total and not stop.is_set():
            n = min(173, n_total - pos)  # awkward chunk size on purpose
            if ring.push(seq[pos:pos + n], stop_event=stop):
                pos += n

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    real_samples = []
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        out, real = _pop(ring, 96)
        real_samples.extend(out[:real].tolist())
        if real == 0:
            if not t.is_alive():
                break  # producer done; remainder (< prebuffer) is gate-stranded by design
            time.sleep(0.0005)  # don't GIL-starve the producer with a hot spin
    stop.set()
    t.join(timeout=5.0)
    got = np.array(real_samples, dtype=np.float32)
    # Strict FIFO: what came out is an exact prefix — nothing lost/reordered/duplicated.
    np.testing.assert_array_equal(got, seq[:got.size])
    # Accounting identity: every pushed sample is either delivered or still in the
    # ring (an unprimed tail < prebuffer stays gated at end-of-stream — documented).
    assert got.size + ring.occupancy == n_total
    assert got.size >= n_total - ring.prebuffer
