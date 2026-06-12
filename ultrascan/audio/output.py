"""Output side of the audio path (DESIGN §3 L0' + the Sim-first twin).

``SpeakerOutput``: sounddevice OutputStream whose callback ONLY drains the SPSC
ring — copy + counter bumps, nothing else (DESIGN §2). 48 kHz mono float32
(= the frozen audifier's FS_OUT). Output opens in shared mode on purpose: the
WASAPI-exclusive rule protects the 250 kHz *input* from the ~48 kHz mix-rate
resample; the audible 48 kHz output has nothing above 24 kHz to lose.

``SimPacedConsumer``: the speakerless twin (DESIGN §1 Sim-first). It drains the
same ring through the same ``pop_into`` API on a wall-clock 48 kHz schedule and
records everything it "played", so continuity and underruns are verifiable
numerically without any audio hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ultrascan.audio.spsc import SpscAudioRing
from ultrascan.dsp.audifier import FS_OUT


class SpeakerOutput:
    """L0' output callback: drain the SPSC ring into the device. Nothing else."""

    def __init__(
        self,
        ring: SpscAudioRing,
        samplerate: float = FS_OUT,
        *,
        device=None,
        blocksize: int = 0,
        record_max_s: float = 0.0,
    ):
        self.ring = ring
        self.samplerate = float(samplerate)
        self.device = device
        self.blocksize = int(blocksize)  # 0 = let PortAudio pick its optimum
        self._stream = None
        self.n_callbacks = 0
        self.n_pa_underflows = 0  # PortAudio-reported, separate from ring.n_underruns
        # Verification capture of what was actually played. Preallocated so the
        # callback stays allocation-free (copy into a fixed buffer only).
        self._rec: Optional[np.ndarray] = None
        self._rec_pos = 0
        if record_max_s > 0:
            self._rec = np.zeros(int(record_max_s * self.samplerate), dtype=np.float32)

    def _callback(self, outdata, frames, time_info, status) -> None:
        # DRAIN ONLY (DESIGN §2): SPSC copy + counters. No FFT / GUI / allocation.
        if status and status.output_underflow:
            self.n_pa_underflows += 1
        out = outdata[:, 0]
        self.ring.pop_into(out)
        rec = self._rec
        if rec is not None and self._rec_pos < rec.size:
            n = min(frames, rec.size - self._rec_pos)
            rec[self._rec_pos:self._rec_pos + n] = out[:n]
            self._rec_pos += n
        self.n_callbacks += 1

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        stream = self._stream
        if stream is not None:
            stream.stop()
            stream.close()
        self._stream = None

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    @property
    def recorded(self) -> np.ndarray:
        """What the callback actually played so far (copy)."""
        if self._rec is None:
            return np.empty(0, dtype=np.float32)
        return self._rec[:self._rec_pos].copy()


class SimPacedConsumer(threading.Thread):
    """Wall-clock-paced ring drain — the hardware-free stand-in for the callback.

    Each wake-up drains exactly the samples a real 48 kHz device would have
    consumed since the last one (elapsed-time accounting, so Windows' coarse
    ~15 ms timer granularity does not distort the schedule). Underrun behaviour
    is therefore identical to the speaker path: a starved ring zero-fills and
    counts, regardless of host timer jitter.
    """

    def __init__(
        self,
        ring: SpscAudioRing,
        samplerate: float = FS_OUT,
        *,
        max_chunk: int = 4096,
    ):
        super().__init__(name="sim-audio-out", daemon=True)
        self.ring = ring
        self.samplerate = float(samplerate)
        self.max_chunk = int(max_chunk)
        self.n_emitted = 0
        self._chunks = []
        self._stop_evt = threading.Event()

    def run(self) -> None:
        t0 = time.perf_counter()
        while not self._stop_evt.is_set():
            due = int((time.perf_counter() - t0) * self.samplerate) - self.n_emitted
            while due > 0:
                n = min(due, self.max_chunk)
                buf = np.empty(n, dtype=np.float32)
                self.ring.pop_into(buf)  # same drain API as the real callback
                self._chunks.append(buf)
                self.n_emitted += n
                due -= n
            self._stop_evt.wait(0.005)

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_evt.set()
        if self.is_alive() and self is not threading.current_thread():
            self.join(timeout=timeout)

    @property
    def output(self) -> np.ndarray:
        """Everything 'played' so far, in order (priming zeros included)."""
        if not self._chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._chunks)
