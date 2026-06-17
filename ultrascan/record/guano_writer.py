"""FROZEN CONTRACT §4.5 — write_event_wav (GUANO).  Frozen at M5.

╔══════════════════════════════════════════════════════════════════════════╗
║  FROZEN at M5.  The FROZEN thing is the SIGNATURE                          ║
║  ``write_event_wav(samples, fs, meta, path) -> None``.  Signature changes  ║
║  need human approval (DESIGN §4, §11).  Later recorders (other container   ║
║  / bit-depth) are NEW functions; this one is not re-edited.                ║
╚══════════════════════════════════════════════════════════════════════════╝

Writes a high-sample-rate **mono 16-bit PCM** WAV at the capture rate (e.g.
250 kHz) with an embedded GUANO ``guan`` metadata chunk (guano-py) — the de-facto
bat-detector format, BTO Acoustic Pipeline compatible.

  * ``samples`` : real float waveform in [-1, 1] (the raw L1 samples). Clipped to
    [-1, 1] and scaled to int16 (the universal bat-WAV depth; float WAV is less
    portable to bat tools).
  * ``fs``      : sample rate (Hz). Written as GUANO ``Samplerate`` (authoritative).
  * ``meta``    : GUANO fields to embed — required ``Timestamp`` / ``Make`` /
    ``Model`` (=UltraMic 250K); optional ``Loc Position`` (GPS), ``Temperature``.
    ``Samplerate`` and ``Length`` are filled from the audio itself (override meta).
  * ``path``    : output .wav path. The CALLER chooses a date/time-stamped name
    (BTO-compatible); this function only writes there.

DESIGN §2: this is a worker-thread / off-callback operation (disk I/O). NEVER call
it from an input/output audio callback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_event_wav(samples: np.ndarray, fs: int, meta: dict, path: str) -> None:
    """High-rate mono 16-bit WAV + embedded GUANO metadata (guano-py)."""
    from guano import GuanoFile  # local import: only the recording milestone needs guano

    x = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if x.size == 0:
        raise ValueError("write_event_wav: refusing to write an empty WAV")
    pcm = np.round(x * 32767.0).astype("<i2")  # int16, little-endian
    fs = int(fs)
    nframes = int(pcm.size)

    gf = GuanoFile()
    gf.wav_params = (1, 2, fs, nframes, "NONE", "not compressed")  # mono, 16-bit
    gf.wav_data = pcm.tobytes()
    gf["GUANO|Version"] = 1.0
    for key, value in (meta or {}).items():
        gf[key] = value
    # Authoritative, from the actual audio — set last so meta cannot contradict them.
    gf["Samplerate"] = fs
    gf["Length"] = round(nframes / float(fs), 6)

    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    gf.filename = str(out)
    gf.write(make_backup=False)
