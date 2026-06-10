"""M1 tests: multi-consumer ring readers (DESIGN §3 — per-consumer cursors)."""

import threading

import numpy as np

from ultrascan.capture.ring_buffer import RingBuffer


def _f32(*vals):
    return np.array(vals, dtype=np.float32)


def test_reader_from_now_sees_only_new_samples():
    rb = RingBuffer(16)
    rb.write(np.arange(4, dtype=np.float32))
    r = rb.reader()  # from_now: backlog invisible
    data, dropped = r.read_new()
    assert data.size == 0 and dropped == 0
    rb.write(_f32(10, 11, 12))
    data, dropped = r.read_new()
    assert np.array_equal(data, _f32(10, 11, 12))
    assert dropped == 0


def test_reader_from_oldest_reads_backlog():
    rb = RingBuffer(16)
    rb.write(np.arange(5, dtype=np.float32))
    r = rb.reader(from_now=False)
    data, dropped = r.read_new()
    assert np.array_equal(data, np.arange(5, dtype=np.float32))
    assert dropped == 0


def test_two_readers_have_independent_positions():
    rb = RingBuffer(32)
    a = rb.reader()
    b = rb.reader()
    rb.write(np.arange(8, dtype=np.float32))
    data_a, _ = a.read_new()
    assert data_a.size == 8
    rb.write(np.arange(8, 16, dtype=np.float32))
    data_a2, _ = a.read_new()
    data_b, _ = b.read_new()
    assert np.array_equal(data_a2, np.arange(8, 16, dtype=np.float32))
    assert np.array_equal(data_b, np.arange(16, dtype=np.float32))  # b saw everything


def test_lapped_reader_reports_dropped_and_resumes_at_oldest():
    rb = RingBuffer(8)
    r = rb.reader()
    rb.write(np.arange(20, dtype=np.float32))  # 20 written into cap-8 ring
    data, dropped = r.read_new()
    assert dropped == 12  # only the last 8 survived
    assert np.array_equal(data, np.arange(12, 20, dtype=np.float32))
    data2, dropped2 = r.read_new()
    assert data2.size == 0 and dropped2 == 0  # caught up afterwards


def test_max_samples_is_strict_fifo_chunking():
    rb = RingBuffer(64)
    r = rb.reader()
    rb.write(np.arange(10, dtype=np.float32))
    c1, d1 = r.read_new(max_samples=4)
    c2, d2 = r.read_new(max_samples=4)
    c3, d3 = r.read_new(max_samples=4)
    assert (d1, d2, d3) == (0, 0, 0)
    assert np.array_equal(np.concatenate([c1, c2, c3]), np.arange(10, dtype=np.float32))


def test_read_spanning_wrap_is_contiguous():
    rb = RingBuffer(8)
    rb.write(np.arange(6, dtype=np.float32))
    r = rb.reader(from_now=False)
    r.read_new()  # consume 0..5
    rb.write(np.arange(6, 12, dtype=np.float32))  # write wraps the physical buffer
    data, dropped = r.read_new()
    assert dropped == 0
    assert np.array_equal(data, np.arange(6, 12, dtype=np.float32))


def test_skip_to_latest_and_lag():
    rb = RingBuffer(32)
    r = rb.reader()
    rb.write(np.arange(10, dtype=np.float32))
    assert r.lag == 10
    assert r.skip_to_latest() == 10
    assert r.lag == 0
    data, dropped = r.read_new()
    assert data.size == 0 and dropped == 0


def test_threaded_producer_no_loss_when_reader_keeps_up():
    """SPMC smoke test: sequence read across threads is gap-free when not lapped."""
    rb = RingBuffer(1 << 14)
    r = rb.reader()
    n_blocks, blocksize = 200, 64
    done = threading.Event()

    def produce():
        for i in range(n_blocks):
            base = i * blocksize
            rb.write(np.arange(base, base + blocksize, dtype=np.float32))
        done.set()

    t = threading.Thread(target=produce)
    received = []
    t.start()
    while not done.is_set() or r.lag > 0:
        data, dropped = r.read_new()
        assert dropped == 0  # huge ring + fast reader: must never be lapped
        if data.size:
            received.append(data)
    t.join()

    seq = np.concatenate(received)
    assert seq.size == n_blocks * blocksize
    assert np.array_equal(seq, np.arange(n_blocks * blocksize, dtype=np.float32))
