# M1 — Display-Only Milestone Report (L0→L1→L2→L4)

## What M1 delivers (DESIGN §6 M1)
- **L1 ring buffer, production version**: single producer + multiple consumers, each
  consumer holds its own read pointer (`RingReader`), with lapped-reader drop detection.
  The M0 snapshot API (`write`/`latest`) is unchanged.
- **L2 DSP worker**: streaming one-sided STFT (`StftStream`, rfft) on its own thread,
  reading via its own ring cursor; columns flow to the GUI through a latest-priority
  bounded deque (display may drop frames — audio will NOT use this path).
- **L4 GUI (pyqtgraph)**: live spectrum (0–125 kHz, dBFS) + horizontally scrolling
  waterfall. Display contrast (`--levels`) is a display knob, separate from audio gain.
- **`spec_audio.render` finalized & FROZEN** (DESIGN §4.1): [256,256] float32 [0,1],
  rfft one-sided, NFFT=512 (rows = bins 0..255), 256 evenly-spaced frames, Hann window,
  DB_DYN_RANGE=60, rows = frequency (row 0 = DC), columns = time. sigscan-shaped for
  v2 CNN reuse. Display waterfall nfft/hop/window are deliberately independent knobs.

## Verification

### Tests
- `pytest -q`: **33 passed** — the 12 pre-existing M0 tests untouched, plus 21 new
  (ring readers / frozen render contract / STFT stream continuity).

### Sim path (saved real capture, `captures/m0_ultramic_keys_250k.wav`)
- `python scripts\m1_view.py --source wav --wav captures\m0_ultramic_keys_250k.wav --blocksize 8192 --duration 8 --screenshot captures\m1_verify_wav.png`
- Waterfall shows the **~25 kHz pest-repeller line** (with its frequency wobble) plus
  broadband key-jingle bursts. 1975 columns in 8 s ≈ 244 cols/s = realtime. 0 drops.
- Sim pacing note: Windows timer granularity (~15 ms) throttles Sim sources below
  realtime when block period < ~15 ms; use `--blocksize 8192` for realtime WAV replay
  (frequency axis is unaffected either way). Real HW does not have this limit.

### Axis calibration (synthetic)
- 45 kHz tone (`--source synthetic --tone-hz 45000`): sharp line at exactly 45 kHz in
  both the live spectrum and the waterfall; full-scale calibration ~0 dBFS verified by
  unit test. Screenshot: `captures/m1_verify_tone45k.png`.

### Real hardware (UltraMic 250K, WASAPI exclusive)
- Device index after reboot: **21** (was 23 — index moves; always re-check with
  `--list-devices`).
- `python scripts\m1_view.py --source wasapi --device 21 --blocksize 256 --duration 8 --screenshot captures\m1_verify_realhw.png`
- **blocks=7858 (≙ 250 kHz realtime), Xrun=0, cols=1947, dropped_samples=0** — with the
  full STFT + GUI load running, blocksize 256 stayed Xrun-free over this 8 s check.
  (Blocksize 256 remains a provisional default per M0; re-measure on long sessions.)
- The ~25 kHz repeller line is visible live on real hardware.

## Bugs found & fixed during M1 (tests + adversarial review)
1. **Ring large-block write broke absolute addressing** — `write()` with a block
   ≥ capacity reset `_write = 0`, violating the "absolute sample i lives at
   buf[i % capacity]" invariant that `RingReader` depends on (`latest()` masked it).
   Caught by the new lapped-reader test; branch rewritten, M0 tests unaffected.
2. **`DspWorker._stop` shadowed `threading.Thread._stop()`** — broke `join()` at
   shutdown (`TypeError: 'Event' object is not callable`). Renamed `_stop_evt`.
3. **Streaming chirp aliased into full-band garbage** (adversarial review, 3/3
   verifiers) — `synth_signal`'s chirp derived its sweep rate from one block span
   but evaluated phase on absolute time, so streamed block-by-block the instantaneous
   frequency passed Nyquist ~1.5 ms in and folded forever; the documented
   `--kind chirp` demo rendered as a solid full-band block. Rewritten as a fixed
   1 s sawtooth sweep (`chirp_period_s`), a function of absolute time only — now
   streaming == single-call at any blocksize. Verified: clean repeating 20→90 kHz
   ramp (`captures/m1_verify_chirp.png`).

4. **`--screenshot` silently ignored without `--duration`** (adversarial review,
   2/3 verifiers) — the screenshot only fired from the timed-quit path. Now also
   taken after a manual window close (idempotent), verified with a close-simulation
   harness.

## Honest notes
- Visual verification was done against *known-signal ground truth* (45 kHz tone,
  20→90 kHz chirp, and the M0 capture whose 25.43 kHz peak was independently
  measured by the M0 offline FFT) — not side-by-side against an external tool
  like Audacity. The frequency-axis calibration checks make this equivalent in
  practice, but a side-by-side look remains a cheap extra check if desired.
- `render()` does not validate non-finite input (NaN/Inf propagates, as reviewed
  and accepted): every in-repo source produces finite float32, and the frozen
  contract documents real-valued samples as the input domain.

## Thread-rule compliance (DESIGN §2)
- Capture callback path = `RingWriter.on_block`: ring memcpy + counters only.
- STFT runs only in `DspWorker` (plain thread, never touches Qt).
- Qt widgets touched only from the GUI thread (QTimer drains the deque).

## Environment note
- PyQt5 5.15 on this machine fails to self-locate its Qt platform plugins;
  `scripts/m1_view.py` sets `QT_QPA_PLATFORM_PLUGIN_PATH` to the venv copy itself.

## Freeze status after M1
- `ultrascan/dsp/spec_audio.py` — **FROZEN** (implemented; signature/behaviour changes
  need human approval; `git diff` must stay empty).
- Still stubs, untouched in M1: `audifier.py`, `gain.py`, `events.py`, `detector.py`,
  `guano_writer.py`.
