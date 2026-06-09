# ultrascan

> **Provisional name** (`ultrascan`) — sister project of **sigscan**. See `ultrascan_DESIGN_and_workorder.md` for the full design (the source of truth).

Realtime ultrasonic spectrum analyzer / bat detector for the **Dodotronic UltraMic 250K** on Windows.
Goal (v1): *see, hear, amplify, and record* ultrasound — continuous live monitoring first.

This repo is built **one milestone at a time** (M0 → M5). Currently: **M0 — capture stress test**.

---

## Status

| Milestone | Scope | State |
|---|---|---|
| **M0** | Capture stress test: WASAPI-exclusive 250k, Xrun monitor, dummy-load, Sim path | ✅ implemented (Sim verified; real-HW judged by Kali) |
| M1 | Display only (live spectrum + waterfall); freeze `spec_audio.render` | ⬜ not started |
| M2 | Continuous heterodyne audification (image-free DDC chain) | ⬜ |
| M3 | Gain / AGC | ⬜ |
| M4 | Event detection + measurement | ⬜ |
| M5 | Event-triggered GUANO recording | ⬜ |

## Environment

- **Python 3.9.13** (matches sigscan exactly → v2 CNN/training harness reuse with no env migration).
- numpy pinned to **1.26.4** (sigscan / torch 2.8.0+cpu compatible; torch is **not** used in v1).

```powershell
# from repo root (PowerShell — run commands one per line, no &&)
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest          # Sim-first unit tests (no hardware needed)
```

> Note: to keep M0 lean, the GUI (`pyqtgraph`/Qt, M1+) and recording (`guano`, M5) deps are
> listed but commented out in `requirements.txt`. M0 only needs numpy / scipy / sounddevice / soxr.

## M0 — capture stress test

The capture pipeline shares one **`InputSource`** abstraction so the exact same stress harness runs on:

- `wasapi`  — real UltraMic via **WASAPI exclusive** mode at 250 kHz (judged on real HW by Kali).
- `synthetic` — known tones/chirps (no hardware). Prints a `SYNTHETIC-ONLY` banner.
- `wav` — replay a WAV file through the same pipe.

Run the stress test (Sim example — 45 kHz tone, 5 s, with a dummy callback load):

```powershell
.\.venv\Scripts\python.exe scripts\m0_stress.py --source synthetic --tone-hz 45000 --duration 5 --load-ms 0
.\.venv\Scripts\python.exe scripts\m0_stress.py --source synthetic --tone-hz 45000 --duration 5 --load-ms 8   # force overruns
# real hardware (Kali):
.\.venv\Scripts\python.exe scripts\m0_stress.py --source wasapi --duration 10 --blocksize 2048
```

It writes a report to `M0_capture_report.md` (achieved latency, tolerable buffer, Xrun results,
and whether energy appears above 24 kHz — Nyquist 125 kHz).

## Layout

```
ultrascan/
├── capture/        # L0 ingest + L1 ring + Sim sources  (M0 lives here)
│   ├── sources.py      InputSource: WASAPI-exclusive / WAV / synthetic
│   ├── ring_buffer.py  minimal single-producer ring (L1 seed)
│   └── stress.py       M0 stress-test core (capture loop, FFT check, report)
├── dsp/            # L2 display DSP + L3 audification + DDC
│   ├── spec_audio.py   FROZEN §4.1 — CNN bridge [256,256] (finalized M1)  ← STUB
│   ├── audifier.py     FROZEN §4.2 — Audifier Protocol (finalized M2)     ← STUB
│   ├── gain.py         FROZEN §4.3 — GainStage Protocol (finalized M3)    ← STUB
│   └── ddc.py          image-free heterodyne DDC chain (M2)               ← STUB
├── detect/         # L5 detection / measurement
│   ├── events.py       Event dataclass §4.4
│   └── detector.py     FROZEN §4.4 — Detector Protocol (finalized M4)     ← STUB
├── record/         # L5 recording
│   └── guano_writer.py FROZEN §4.5 — write_event_wav (finalized M5)       ← STUB
└── gui/            # L4 pyqtgraph (M1+)                                    ← placeholder
```

> **Freeze rule:** the frozen-contract files are signature **stubs** in M0. They are finalized and
> frozen at the milestone noted (M1+); from then on their `git diff --stat` must stay empty.
