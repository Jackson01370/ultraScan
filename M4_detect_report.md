# M4 — Event Detection + Measurement Report (DESIGN §4.4 / §6 M4)

## Scope
First milestone that **observes** the signal rather than processing it for output.
A `Detector` (method B: adaptive noise-floor SNR) runs as a **third, independent
L1 consumer** hung off the display side (DESIGN §3 two-pass separation), draws
event boxes on the M1 waterfall, and optionally logs measurements to CSV. The
**audio path (M2b/M3) is untouched** — `git diff` empty for `spec_audio.py`,
`audifier.py`, `gain.py`, and the M5 record stub. Recording stays a stub (M5).

## Frozen at M4 (DESIGN §4.4)
- **`detect/events.py` — `Event`**: field set/types frozen (the §4.4 contract).
  `duration` is derived (`t_end-t_start`), `slope` is kHz/ms, `n_pulses`/`ipi`
  reserved (M4 emits one time-segment → `n_pulses=1`, `ipi=None`).
- **`detect/detector.py` — `Detector` Protocol** (`detect(frame_or_buffer) ->
  List[Event]`) + the M4 implementation `AdaptiveSnrDetector`. Same discipline as
  M2a/M3: the single impl lands here, then the file freezes; later detectors
  (e.g. a CNN) are NEW classes/modules. The freeze is on the **signature**;
  tuning `AdaptiveSnrDetector`'s default params is allowed with approval (gain.py
  scope rule). The banner states this.

## `AdaptiveSnrDetector` — method B (adaptive, not fixed, threshold)
Per dBFS STFT column (`StftStream.push` output):
1. **Per-bin background EMA** (in dB). A bin exceeding its background by
   `snr_threshold_db` (default 12) is *active*. The background **adapts**, so a
   steady tone (the ~25 kHz pest-repeller) is absorbed → never detected. This is
   the whole point: an adaptive threshold ignores the repeller; a fixed one
   could not.
2. **Anti-self-cancel / anti-stuck**: active bins adapt *slowly*
   (`bg_tau_active_s`=3 s) so a transient event doesn't pull up its own
   background mid-event, yet a *persistent* new tone is still absorbed after a few
   seconds (not flagged forever).
3. **Noise rejection by concentration + persistence** (the key to not firing on
   the noise floor, whose per-bin dB spikes are large and scattered): a frame's
   activity is the **strongest contiguous run** of ≥`min_run_bins` (default 4)
   active bins — scattered single-bin noise never forms one. Runs are tracked
   across frames by frequency overlap; a track shorter than `min_event_s`
   (default 50 ms) is discarded (a blip), a lasting one is emitted.
4. **Measurement** (robust to onset/offset window-transient smear): `f_peak` =
   loudest bin; frequency extent = **peak-frequency excursion** (gives a chirp
   its swept range, a tone ~0) ∪ the **−20 dB spectral width at the single
   strongest frame** (a steady frame, never a transient); `slope` = least-squares
   peak-freq-vs-time (kHz/ms); `snr_db` = peak SNR over background.

Accepted modelling limits (documented in code): **dominant source per instant**
(a strictly simultaneous weaker source isn't separately tracked — sequential
events at different freqs ARE); bandwidth is the −20 dB extent (window-limited,
so a pure tone reads ~one bin pair, not literally 0); very short pulses
(bat-click ms) full capture is deferred to future tuning (waterfall-hop
granularity here).

## Plumbing (all NEW files; no frozen/audio file touched)
| File | Role |
|---|---|
| `detect/detector.py` | `Detector` Protocol + `AdaptiveSnrDetector` (FROZEN). |
| `detect/worker.py` | `DetectorWorker` — own L1 `RingReader` + own `StftStream`; resets STFT+detector on a reader gap (like `DspWorker`); thread-safe `snapshot()` for the GUI; optional `event_sink` (logger) called on the worker thread. |
| `detect/event_log.py` | `EventCsvLogger` — one row per event, flushed per row; the **measurement** log (NOT audio). |
| `gui/event_view.py` | `EventOverlayView(BandSelectView)` — event boxes+labels on the waterfall, GUI-thread only; inherits band drag-to-listen. |
| `scripts/m4_detect.py` | CLI: live GUI (boxes + drag-to-listen) / `--no-gui` headless; `--log-events PATH` (default OFF); all detector params exposed; reuses the M2b/M3 audio path (optional, `--no-audio`). |

## Verification — Sim first (numeric)
`pytest -q`: **94 passed, 1 skipped** — the 78 pre-existing **untouched**, +16
new (the 1 skip is still the gitignored real-capture WAV). New tests
(`test_detector.py` ×13, `test_detect_pipeline.py` ×3), all through the real
`StftStream`:

**Repeller-ignore (the heart of method B):**
- Steady 25 kHz repeller + noise, 3 noise levels (1e-3 / 2e-3 / 3e-3): **0
  events** every time — the line is absorbed.
- Repeller + 50 kHz burst → **exactly 1 event** at 50.05 kHz; 25 kHz is NOT in
  any event's range.
- Detection **unaffected by repeller on/off**: the same burst measures the same
  with the repeller present and absent.

**Measurement accuracy** (bin=122 Hz, hop=4.10 ms):
| Signal | f_peak | duration | bandwidth | slope |
|---|---|---|---|---|
| 50 kHz tone, 0.5 s | 50049 Hz (≤1 bin err) | 0.508 s | 366 Hz (narrow) | ~0 |
| 40→60 kHz chirp, 0.5 s | — | 0.51 s | 19.9 kHz | **0.0400 kHz/ms** (true 0.040) |
| two sequential bursts 40 k / 70 k | 40039 / 69946 | — | narrow | ~0 → **2 events** |

Plus: empty/noise-only input → no events; `reset()` clears state but keeps the
time cursor; `finalize()` emits an event still open at stream end; short blip
(15 ms < min-event) rejected; constructor validation; `Detector` protocol
conformance; CSV log round-trip.

**End-to-end CLI (headless Sim):** `m4_detect --source wav` on a generated scene
(repeller 25 k + 50 k@[1.0,1.5] + 70 k@[2.5,2.9] + noise):
```
detected 2 event(s):
  t=[1.00,1.51]s dur=508ms f_peak=50.05kHz bw=0.37kHz snr=69.8dB
  t=[2.50,2.90]s dur=406ms f_peak=69.95kHz bw=0.37kHz snr=68.9dB
in_xruns=0 det_dropped_samples=0  event log -> captures\m4_events.csv (2 rows)
```
Repeller ignored, both bursts detected+measured+logged, 0 xrun / 0 drop.

**GUI overlay:** offscreen smoke test builds `EventOverlayView` and runs its
`_refresh()` render path with 2 events → 2 boxes + 2 labels, no error. The
on-screen screenshot + the live repeller test are **Kali's real-HW judgment**.

## Thread-rule compliance (DESIGN §2)
Detector runs only on its own worker thread (numpy STFT releases the GIL); never
in a callback. Qt touched only on the GUI thread (QTimer reads the worker's
thread-safe snapshot). Audio input/output callbacks unchanged. The frozen
audifier/gain are not touched by the detector at all (separate L1 cursor).

## Honest notes / accepted limits
- **SYNTHETIC-ONLY here**: all numbers Sim. Real-HW (UltraMic, WASAPI; device
  index via `--list-devices` each time — was 23/21, moves) verification — the
  repeller staying un-boxed while keys/other sounds get boxed live, and overlay
  alignment by eye — is **Kali's** (DESIGN §6 roles).
- Detector output is a **rule/heuristic** result → **NOT CNN training labels**
  (DESIGN §11), stated in the banner, CLI, and module docs.
- A repeller that switches ON *mid-stream* is detected as a finite onset event
  (~few s, scales with loudness) then absorbed — the real repeller is always-on
  (absorbed at init), so this is benign; tune `--bg-tau-active-s` if needed.
- **venv is Python 3.11.9** (no torch in v1; v2/CNN concern only).

## Freeze status after M4
- `spec_audio.py`, `audifier.py`, `gain.py` — FROZEN, diff empty this milestone.
- **`detect/events.py`, `detect/detector.py` — NEWLY FROZEN at M4.**
- Still a stub: `record/guano_writer.py` → M5 (diff empty).
