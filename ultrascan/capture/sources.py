"""Input source abstraction (DESIGN §1 "Sim-first", §6 M0).

One interface, three backends, so the *same* M0 stress harness runs with or
without hardware:

  - WasapiExclusiveSource : real UltraMic 250K via WASAPI **exclusive** mode.
  - SyntheticSource       : known tones/chirps, no hardware  (SYNTHETIC-ONLY).
  - WavFileSource         : replay a WAV through the same pipe (SYNTHETIC-ONLY).

Contract: a source pushes mono float32 blocks to ``on_block(block, status)``.
The callback is the only place hardware timing matters, so per DESIGN §2 it must
stay copy-only — no FFT / GUI / heavy work. The Sim sources reproduce overrun
behaviour (a slow consumer -> ``input_overflow``) so the dummy-load boundary test
in §6 M0 can be exercised without a microphone.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

OnBlock = Callable[[np.ndarray, "CaptureStatus"], None]

# UltraMic 250K native rate; Nyquist = 125 kHz (DESIGN §1).
DEFAULT_RATE = 250_000.0
DEFAULT_BLOCKSIZE = 2048


@dataclass
class CaptureStatus:
    """Per-block health, normalised across backends."""

    input_overflow: bool = False
    input_underflow: bool = False
    detail: str = ""


# ── signal generation (shared by SyntheticSource and tests) ────────────────
def synth_signal(
    n: int,
    samplerate: float = DEFAULT_RATE,
    *,
    start_index: int = 0,
    kind: str = "tone",
    tone_hz: float = 45_000.0,
    chirp_hz: tuple = (20_000.0, 90_000.0),
    amplitude: float = 0.5,
) -> np.ndarray:
    """Deterministic real-valued test signal (float32), indexed by absolute sample.

    ``start_index`` keeps phase continuous across successive blocks. A 45 kHz tone
    at 250 kHz is the canonical M0 probe: it must show up as energy well above the
    24 kHz "share mode would have eaten this" line.
    """
    idx = np.arange(start_index, start_index + n, dtype=np.float64)
    t = idx / samplerate
    if kind == "tone":
        sig = amplitude * np.sin(2.0 * np.pi * tone_hz * t)
    elif kind == "chirp":
        f0, f1 = chirp_hz
        period = max(n, 1) / samplerate  # one sweep across this block span
        k = (f1 - f0) / period
        phase = 2.0 * np.pi * (f0 * t + 0.5 * k * (t ** 2))
        sig = amplitude * np.sin(phase)
    else:
        raise ValueError(f"unknown synthetic kind: {kind!r}")
    return sig.astype(np.float32)


class InputSource(ABC):
    """Common interface for every capture backend."""

    name: str = "input"
    is_synthetic: bool = False

    def __init__(self, samplerate: float, blocksize: int, channels: int = 1):
        self.samplerate = float(samplerate)
        self.blocksize = int(blocksize)
        self.channels = int(channels)

    @abstractmethod
    def start(self, on_block: OnBlock) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def is_running(self) -> bool: ...

    def describe(self) -> dict:
        return {
            "name": self.name,
            "is_synthetic": self.is_synthetic,
            "samplerate": self.samplerate,
            "blocksize": self.blocksize,
            "channels": self.channels,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


# ── Sim sources (no hardware) ──────────────────────────────────────────────
class _ThreadedSource(InputSource):
    """Realtime block emitter for Sim sources, with overrun simulation.

    Emits blocks on a wall-clock schedule (period = blocksize / samplerate). A block
    is flagged ``input_overflow`` when the *previous* ``on_block`` call took longer
    than one block period to return — mirroring a real device whose ring overflows
    while the callback is blocked. This keys overrun to consumer cost only, so it is
    independent of host ``time.sleep`` timer jitter (coarse on Windows, ~15 ms).
    """

    is_synthetic = True

    def __init__(self, samplerate: float, blocksize: int, channels: int = 1):
        super().__init__(samplerate, blocksize, channels)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False

    @abstractmethod
    def _produce(self, idx: int, n: int) -> Optional[np.ndarray]:
        """Return the next block, or None when the source is exhausted."""

    def start(self, on_block: OnBlock) -> None:
        if self._running:
            raise RuntimeError("source already started")
        self._stop.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, args=(on_block,), name=f"{self.name}-emit", daemon=True
        )
        self._thread.start()

    def _run(self, on_block: OnBlock) -> None:
        period = self.blocksize / self.samplerate
        next_emit = time.perf_counter()
        idx = 0
        prev_dt = 0.0  # duration of the previous on_block call
        try:
            while not self._stop.is_set():
                block = self._produce(idx, self.blocksize)
                if block is None:
                    break
                idx += block.size
                overflow = prev_dt > period  # previous callback overran the budget
                status = CaptureStatus(
                    input_overflow=overflow,
                    detail=f"sim consumer={prev_dt * 1e3:.2f}ms > period" if overflow else "",
                )
                c0 = time.perf_counter()
                on_block(block, status)
                prev_dt = time.perf_counter() - c0
                next_emit += period
                slack = next_emit - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif -slack > period:
                    next_emit = time.perf_counter()  # fell far behind; resync clock
        finally:
            self._running = False

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._running = False
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running


class SyntheticSource(_ThreadedSource):
    """Endless synthetic tone/chirp at the native rate (no hardware)."""

    name = "synthetic"

    def __init__(
        self,
        samplerate: float = DEFAULT_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        *,
        tone_hz: float = 45_000.0,
        kind: str = "tone",
        amplitude: float = 0.5,
        chirp_hz: tuple = (20_000.0, 90_000.0),
    ):
        super().__init__(samplerate, blocksize, channels=1)
        self.tone_hz = float(tone_hz)
        self.kind = kind
        self.amplitude = float(amplitude)
        self.chirp_hz = chirp_hz

    def _produce(self, idx: int, n: int) -> Optional[np.ndarray]:
        return synth_signal(
            n,
            self.samplerate,
            start_index=idx,
            kind=self.kind,
            tone_hz=self.tone_hz,
            chirp_hz=self.chirp_hz,
            amplitude=self.amplitude,
        )

    def describe(self) -> dict:
        d = super().describe()
        d.update(kind=self.kind, tone_hz=self.tone_hz, amplitude=self.amplitude)
        return d


class WavFileSource(_ThreadedSource):
    """Replay a WAV file (mono float32) through the same pipe. Stops at EOF."""

    name = "wav"

    def __init__(
        self,
        path: str,
        blocksize: int = DEFAULT_BLOCKSIZE,
        *,
        loop: bool = False,
    ):
        from scipy.io import wavfile  # local import: only needed for WAV replay

        rate, data = wavfile.read(path)
        data = self._to_mono_float32(data)
        super().__init__(float(rate), blocksize, channels=1)
        self.path = path
        self.loop = loop
        self.data = data

    @staticmethod
    def _to_mono_float32(data: np.ndarray) -> np.ndarray:
        if data.ndim > 1:
            data = data[:, 0]
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = max(abs(info.min), info.max)
            data = data.astype(np.float32) / float(scale)
        else:
            data = data.astype(np.float32)
        return data

    def _produce(self, idx: int, n: int) -> Optional[np.ndarray]:
        total = self.data.size
        pos = idx % total if self.loop else idx
        if not self.loop and pos >= total:
            return None
        end = pos + n
        if end <= total:
            return self.data[pos:end].copy()
        if self.loop:
            head = self.data[pos:]
            tail = self.data[: end - total]
            return np.concatenate([head, tail])
        return self.data[pos:total].copy()  # last partial block

    def describe(self) -> dict:
        d = super().describe()
        d.update(path=self.path, loop=self.loop, n_samples=int(self.data.size))
        return d


# ── real hardware (judged on real HW by Kali) ──────────────────────────────
class WasapiExclusiveSource(InputSource):
    """Real UltraMic 250K via WASAPI **exclusive** mode (sounddevice / PortAudio).

    Shared mode is forbidden (DESIGN §2/§11): it resamples to the system mix rate
    (~48 kHz) and everything above 24 kHz is destroyed. The sounddevice callback
    here is strictly copy-only.
    """

    name = "wasapi"
    is_synthetic = False

    def __init__(
        self,
        samplerate: float = DEFAULT_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        *,
        device=None,
        channels: int = 1,
    ):
        super().__init__(samplerate, blocksize, channels)
        self.device = device
        self._stream = None
        self._on_block: Optional[OnBlock] = None

    def verify_native_rate(self) -> dict:
        """Check the device can be opened EXCLUSIVE at the requested native rate.

        Returns a dict for the M0 report. Never raises — a failure here is itself a
        result Kali needs to see (e.g. driver refuses 250k exclusive).
        """
        import sounddevice as sd

        out = {"requested_rate": self.samplerate, "ok": False, "error": None}
        try:
            extra = sd.WasapiSettings(exclusive=True)
            sd.check_input_settings(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                dtype="float32",
                extra_settings=extra,
            )
            info = sd.query_devices(self.device, "input")
            out.update(
                ok=True,
                device_name=info.get("name"),
                device_default_rate=info.get("default_samplerate"),
            )
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def _sd_callback(self, indata, frames, time_info, status):
        # COPY ONLY (DESIGN §2). Heavy work happens off-thread in the consumer.
        block = np.array(indata[:, 0], dtype=np.float32, copy=True)
        st = CaptureStatus(
            input_overflow=bool(getattr(status, "input_overflow", False)),
            input_underflow=bool(getattr(status, "input_underflow", False)),
            detail=str(status) if status else "",
        )
        if self._on_block is not None:
            self._on_block(block, st)

    def start(self, on_block: OnBlock) -> None:
        import sounddevice as sd

        self._on_block = on_block
        extra = sd.WasapiSettings(exclusive=True)
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="float32",
            channels=self.channels,
            device=self.device,
            extra_settings=extra,
            callback=self._sd_callback,
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

    def describe(self) -> dict:
        d = super().describe()
        d.update(device=self.device, mode="WASAPI-exclusive")
        return d


def make_source(backend: str, **kwargs) -> InputSource:
    """Factory: 'synthetic' | 'wav' | 'wasapi'.

    First arg is the backend name; ``kind`` is forwarded as a kwarg to
    SyntheticSource (tone/chirp), so the names are kept distinct on purpose.
    """
    backend = backend.lower()
    if backend == "synthetic":
        return SyntheticSource(**kwargs)
    if backend == "wav":
        return WavFileSource(**kwargs)
    if backend == "wasapi":
        return WasapiExclusiveSource(**kwargs)
    raise ValueError(f"unknown source backend: {backend!r} (use synthetic|wav|wasapi)")
