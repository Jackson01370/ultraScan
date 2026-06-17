# M5 — Event-Triggered GUANO Recording Report (DESIGN §4.5 / §6 M5) — v1 FINAL

## Scope
The last v1 milestone: M4's detected events trigger a **pre-rolled, high-rate
(250 kHz) GUANO WAV** write of the raw L1 samples. Recording hangs off the
detection (display-side) path; the **audio listen path (M2b/M3) is untouched**.
Recording is OFF by default and event-only. Frozen-5 (`spec_audio`, `audifier`,
`gain`, `detector`, `events`) `git diff` **empty**.

## Frozen at M5 (DESIGN §4.5)
- **`record/guano_writer.py` — `write_event_wav(samples, fs, meta, path)`**: the
  signature is frozen (banner states the scope rule). Writes **mono 16-bit PCM**
  (float [-1,1] → int16; the universal bat-WAV depth, BTO-compatible) at the full
  capture rate with an embedded GUANO `guan` chunk (guano-py). `Samplerate`
  (=`fs`) and `Length` (from sample count) are authoritative; `Timestamp` /
  `Make` / `Model` (+ optional GPS / Temperature) come from `meta`.

## Pre-trigger (the heart of M5)
The detector only flags an event a little AFTER it begins, so recording from
"now" would clip the onset. Instead the recorder reads **back** from L1 — the
truth source doubles as the pre-trigger ring buffer (DESIGN §6 M5) — via the new
`RingBuffer.read_absolute(start_abs, n)` (random-access, clamped to resident).
An event's detector-relative time maps to an absolute L1 sample as
`l1_start_abs + t·fs` (`DetectorWorker.l1_start_abs` = where detection began).
The window is `[onset − preroll, end + postroll]`.

## Plumbing (new / additive; no frozen or audio file touched)
| File | Change |
|---|---|
| `record/guano_writer.py` | implemented + FROZEN `write_event_wav` (guano-py). |
| `record/recorder.py` (new) | `EventRecorder` — a `DetectorWorker` `event_sink`; on each event, pre-roll grab from L1 + `write_event_wav`. Runs on the detector worker thread (disk I/O off the audio callback, §2). BTO-style filename `YYYYMMDD_HHMMSS_mmm_<peak>kHz.wav`. |
| `capture/ring_buffer.py` | **added** `read_absolute` (additive; existing ring behaviour/tests unchanged). |
| `detect/worker.py` | **added** `l1_start_abs` (the time→sample anchor). |
| `scripts/m4_detect.py` | **added** `--record` / `--record-dir` / `--preroll-ms` / `--postroll-ms` / `--max-record-s` (recording OFF by default; the `--log-events` CSV still works alongside). |
| `requirements.txt` | uncommented `guano>=1.0`; installed `guano-1.0.16` into `.venv`. |

## Verification — Sim first (numeric)
`pytest -q`: **99 passed, 1 skipped** (94 pre-existing untouched + 5 new in
`test_record.py`; the skip is still the gitignored real-capture WAV; 2 warnings
are scipy noting the extra `guan` chunk — harmless):
- `write_event_wav` round-trip: reads back **250 kHz int16**, peak at 50 kHz,
  GUANO `Samplerate`=250000 / `Make` / `Model` / `Timestamp` (datetime) /
  `Length` correct; empty input rejected.
- `read_absolute` clamps to resident (front-overwritten and tail-past-head).
- **PRE-TRIGGER**: noise→burst scene (no repeller), record triggered → the
  burst onset lands ~preroll INTO the file (`0.2 < onset < 0.45 s` for 0.3 s
  preroll), file length ≈ preroll+event+postroll, lead-in > 0.2 s.
- **EVENT-ONLY**: steady repeller → **0** WAVs; repeller+burst → **1** WAV.

**End-to-end CLI (headless Sim)** — `m4_detect --source wav --record` on the
repeller+2-burst scene:
```
events=2  in_xruns=0  det_dropped_samples=0
  t=[1.00,1.51]s f_peak=50.05kHz ...   t=[2.50,2.90]s f_peak=69.95kHz ...
recordings -> captures/m5_test (2 WAV(s))   [repeller (25 kHz) triggered nothing]
```
Read back: both WAVs **250 kHz**, GUANO `Make=Dodotronic Model="UltraMic 250K"`
`Samplerate=250000` `Timestamp=…` `Note="…preroll=300ms"`. The **50 kHz / 70 kHz
band rises at 297 ms** into each file → the ~300 ms preroll is present, **onset
not clipped**. (The raw recording also contains the continuous 25 kHz repeller —
correct: it is a faithful raw capture; only the *trigger* ignores the repeller.)

## Honest notes / accepted limits
- **Preroll guaranteed; postroll best-effort.** Preroll is in the past → always
  resident in L1 (provided the ring ≥ event+preroll; default 4 s covers events to
  ~3 s). Postroll is clamped to whatever L1 has written by trigger time — in the
  Sim run it came out ~70 ms of the requested 200 ms. The onset (the work order's
  priority) is never clipped.
- **16-bit PCM** (float→int16) for BTO/GUANO tool compatibility; raw float was not
  retained (documented in the banner).
- `read_absolute` copies under the ring lock — a bounded memcpy (≤`max-record-s`),
  comparable to the existing streaming reads (M2b's 180 s soak was Xrun-free);
  real-HW Xrun under record load is Kali's check (§2).
- Time→sample mapping assumes the detector kept up (no L1 gap) — exact in all Sim
  runs; a sustained-overload gap offsets subsequent events (documented).
- **Quarantine folder** (`captures/_review_pending/`, DESIGN §6) is deferred — not
  in the M5 work-order scope (no scope creep).
- **SYNTHETIC-ONLY here.** Real-HW (UltraMic, WASAPI; device via `--list-devices`)
  — repeller ignored, keys/etc. write a WAV, replay the WAV via `--source wav` —
  is **Kali's** (DESIGN §6 roles). venv is Python 3.11.9 (no torch in v1).

## v1 COMPLETE (M0–M5)
Ultrasound is now **seen** (M1 spectrum + waterfall), **heard** (M2 image-free
heterodyne audification), **amplified** (M3 AGC), **detected & measured** (M4
adaptive-SNR events), and **recorded** (M5 pre-rolled GUANO WAVs). The whole chain
shares one L1 truth source with physically separated display / audio / detection
consumers (DESIGN §3), and 0-Xrun real-HW behaviour was established at M2b.

## v2 seams left intact (per design §4 / §7)
- **CNN bridge**: `spec_audio.render` → [256,256] float32 [0,1] is frozen and
  sigscan-shaped; detected events can be rendered through it to reuse sigscan's
  CNN/training harness unchanged. **Rule-detector output is NOT used as training
  labels** (enforced in banners/docs; label-noise prevention) — the seam exists,
  the labels do not yet.
- **Frozen contracts** ready for swap-in: `Audifier` (→ `TimeExpansionAudifier`),
  `GainStage` (→ `Normalize`/`Compressor`), `Detector`/`Event`,
  `write_event_wav`. New implementations are new classes/modules; signatures hold.
- Deferred by design: time-expansion audifier, ultrasonic band-plan, GPS/GSI map,
  measurement cursor, noise reduction, dB SPL calibration, event quarantine.

## Freeze status after M5
- `spec_audio.py`, `audifier.py`, `gain.py`, `detect/detector.py`,
  `detect/events.py` — FROZEN, diff empty this milestone.
- **`record/guano_writer.py` — NEWLY FROZEN at M5.** No remaining stubs — v1 done.
