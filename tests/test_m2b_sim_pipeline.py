"""M2b end-to-end Sim acceptance (DESIGN §1 Sim-first, no audio hardware):

    SyntheticSource (realtime pacing) -> L1 ring -> AudioWorker (frozen DDC)
        -> SPSC ring -> SimPacedConsumer (wall-clock 48 kHz drain)

Pins the M2b完了基準 numerically: zero ring underruns after priming, and the
"played" stream (priming silence stripped) is exactly the FIFO prefix of what
the DDC produced — i.e. the output is continuous, no zero-fill gaps inside.
"""

import time

import numpy as np

from ultrascan.audio.output import SimPacedConsumer
from ultrascan.audio.spsc import SpscAudioRing
from ultrascan.audio.worker import AudioWorker
from ultrascan.capture.ring_buffer import RingBuffer
from ultrascan.capture.sources import SyntheticSource, synth_signal
from ultrascan.dsp.audifier import HeterodyneAudifier
from ultrascan.gui.pipeline import RingWriter

FS_IN = 250_000.0
FS_OUT = 48_000.0
RUN_S = 3.0


def test_sim_pipeline_runs_continuously_without_underruns():
    # blocksize 8192: realtime-capable under Windows' ~15 ms timer (M1 finding).
    source = SyntheticSource(FS_IN, 8_192, tone_hz=45_000.0)
    l1 = RingBuffer(int(4 * FS_IN))
    writer = RingWriter(l1)
    spsc = SpscAudioRing(capacity=int(2 * FS_OUT), prebuffer=8_192)  # ~170 ms headroom
    worker = AudioWorker(
        l1.reader(), HeterodyneAudifier(), spsc, FS_IN,
        f_lo_sel=40_000.0, bandwidth=10_000.0,
    )
    consumer = SimPacedConsumer(spsc, FS_OUT)

    worker.start()
    consumer.start()
    source.start(writer.on_block)
    time.sleep(RUN_S)
    # Snapshot the verdict counters BEFORE teardown: stopping the source starves
    # the chain by design, and that trailing starvation is not the measurement.
    underruns = spsc.n_underruns
    popped_real = spsc.n_popped_real
    popped_zero = spsc.n_popped_zero  # with 0 underruns these are ALL priming zeros
    source.stop()
    worker.stop()
    consumer.stop()

    played = consumer.output

    # 1) the queue never underran while running (M2b完了基準: アンダーランゼロ)
    assert underruns == 0
    # 2) the consumer actually played a realtime-sized stream of real samples
    assert popped_real > 0.7 * RUN_S * FS_OUT
    # 3) continuity: with 0 underruns, `played` is [priming zeros][real FIFO...],
    #    so count-based slicing recovers the exact FIFO prefix of the DDC output
    #    for the exact input consumed (no zero-fill gaps inside).
    body = played[popped_zero:popped_zero + popped_real]
    ref_aud = HeterodyneAudifier()
    ref_aud.configure(40_000.0, 10_000.0, FS_IN)
    ref = ref_aud.process(synth_signal(worker.n_in_samples, FS_IN, tone_hz=45_000.0))
    assert ref.size >= body.size
    np.testing.assert_allclose(body, ref[:body.size], atol=1e-5)
    # 4) and it is the expected sound: 45 kHz mixed down by LO 40 kHz -> 5 kHz.
    spec = np.abs(np.fft.rfft(body * np.hanning(body.size)))
    peak_hz = np.argmax(spec) * FS_OUT / body.size
    assert abs(peak_hz - 5_000.0) < 50.0
    # 5) input side stayed healthy too
    assert writer.n_xruns == 0
    assert worker.n_dropped_in == 0
