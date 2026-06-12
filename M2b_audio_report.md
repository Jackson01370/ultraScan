# M2b — Live Audio Output Report (L3 worker + SPSC queue + output callback + band drag)

## Scope
Second half of M2 (DESIGN §6 M2 / §3 L3 / §5): the frozen `HeterodyneAudifier`
(M2a) wired into a real-time audio output path. Frozen files untouched
(`spec_audio.py`, `audifier.py` — `git diff` empty). `gain.py` / detect / record
remain stubs (gain/AGC is M3).

## Click policy — decided explicitly: **(a)**
M2a established that `configure()` fully resets DDC state, so band re-selection
is not click-free. **M2b accepts the click (policy (a)): continuous sound has
priority.** The same applies to an L1 input gap (lapped audio reader): the
transient passes through. Crossfade/ramp smoothing is recorded as an **M2c+
item and deliberately NOT implemented here.** Underrun recovery likewise
inserts an audible gap and re-primes (see SPSC below) — same policy.

## Implementation (all new files; no frozen file touched)
| File | Role |
|---|---|
| `ultrascan/audio/spsc.py` | `SpscAudioRing` — bounded strict-FIFO float32 ring. Producer `push()` blocks (backpressure); consumer `pop_into()` NEVER blocks: shortfall is zero-filled + counted. **Priming**: silence until `prebuffer` samples queued (先読み headroom, DESIGN §2); start-of-stream priming is not an underrun; a primed-but-short pop IS (`n_underruns`), and the ring re-primes to rebuild headroom. |
| `ultrascan/audio/worker.py` | `AudioWorker` — L3 thread: **own `RingReader` cursor on L1** (separate consumer from the display `DspWorker` — DESIGN §1 two-pass separation), drives the frozen audifier single-threadedly, pushes to SPSC. `request_band()` is asynchronous: applied between blocks; a refused selection (M2a atomic configure) keeps the old band playing and surfaces the error string. `volume` is a fixed safety attenuator applied on this thread — explicitly NOT the M3 GainStage. |
| `ultrascan/audio/output.py` | `SpeakerOutput` — L0' sounddevice OutputStream; callback = `pop_into` drain + counters only (copy-only rule §2; record capture writes into a **preallocated** buffer). Output opens shared-mode on purpose: WASAPI-exclusive protects the 250k *input*; the 48 kHz output has nothing above 24 kHz to lose. `SimPacedConsumer` — speakerless twin draining the same API on a wall-clock 48 kHz schedule (elapsed-time accounting, immune to Windows' ~15 ms timer). |
| `ultrascan/gui/band_view.py` | `BandSelectView(LiveView)` — horizontal `LinearRegionItem` on the M1 waterfall; drag → lower edge = `f_lo_sel`, height = `bandwidth` → `request_band()`. Refusals show in the status bar (REFUSED: …), no crash. Bounds clamped to [0, Nyquist]. |
| `scripts/m2b_listen.py` | CLI: GUI+speaker / `--no-gui` headless / `--sim-out` speakerless. Prints a counter report (snapshot taken BEFORE teardown) + offline FFT/continuity analysis of what was actually played; `--save-out` writes the played 48 kHz WAV. |

## Thread-rule compliance (DESIGN §2)
- Input callback: unchanged M0/M1 path (ring memcpy + counters).
- Output callback: `ring.pop_into(outdata)` + fixed-buffer record copy + counter
  bumps. No FFT / GUI / allocation / waiting.
- Qt only on the GUI thread; band drag crosses threads via `request_band()`
  (lock-protected pending slot, applied by the worker thread).
- Frozen audifier is touched by exactly one thread (the worker).

## Verification — Sim first (no hardware, numeric)
`--sim-out --no-gui`, wall-clock-paced 48 kHz drain, prebuffer 3840 (80 ms):

| Run | Result |
|---|---|
| Synthetic 45 kHz tone, LO 40 kHz, 5 s | queue underruns **0**, peak **5000.1 Hz**, rms **0.3535** (= M2a's pinned 0.5/√2), max zero-run **0** |
| Real capture `m0_ultramic_keys_250k.wav`, LO 20 kHz, 8 s | queue underruns **0**, peak **5427.7 Hz** (= M0's 25.43 kHz repeller − 20 kHz; M2a offline: 5430.2), max zero-run **0** |

Continuity is proven count-based, not by ear: with 0 underruns the played
stream is `[priming zeros][exact FIFO prefix of DDC output]`, and the
end-to-end test asserts sample-exact equality (atol 1e-5) against a one-shot
reference run of the same frozen audifier over the same input.

## Verification — real hardware (UltraMic 250K, device 21, WASAPI-exclusive 250k)
Real speakers (48 kHz shared), blocksize 256, LO 20 kHz / W 10 kHz; the room's
pest repeller (~25.4 kHz) as the live source. Audible on speakers as ~5.4 kHz.

| Run | in-callbacks | in-Xrun | out-callbacks | queue underruns | PA out-underflows | played peak |
|---|---:|---:|---:|---:|---:|---:|
| Smoke 10 s (headless) | 9 761 | **0** | 385 | **0** | **0** | 5430.6 Hz |
| **Soak 180 s** (headless) | 175 956 | **0** | 6 930 | **0** | **0** | 5432.7 Hz |
| GUI 20 s (display STFT + waterfall + audio + speakers simultaneously) | 19 511 | **0** | 768 | **0** | **0** | 5432.2 Hz |

- **The M0/M1 homework is closed**: blocksize 256 was "provisional, re-measure
  under audio I/O load". With the full chain live (250k exclusive capture +
  DDC + 48k speaker output (+ GUI in the third run)), **3 minutes → 0 Xrun /
  0 underrun on every counter**. 256 stands as the default.
- Soak accounting: `popped_zero = 4992` exactly equals the initial priming fill
  in all runs → zero mid-stream zero-fill. (`max_zero_run=1` in the soak body is
  a single sample whose value is exactly 0.0 in 8.6 M real samples — fills only
  occur in callback-sized runs, and the underrun counter is 0.)
- Screenshot `captures/m2b_gui_band.png`: band region 20–30 kHz drawn over the
  live waterfall with the repeller line inside it; status bar shows
  `audio 20.0–30.0 kHz buf .../96000 underruns 0`.
- Artifacts: `captures/m2b_sim_tone.wav`, `m2b_sim_repeller.wav`,
  `m2b_realhw_smoke.wav`, `m2b_realhw_soak180.wav`, `m2b_gui_out.wav` (all
  gitignored with the rest of `captures/`).

## Tests
`pytest -q`: **65 passed** — 50 pre-existing **untouched**, 15 new:
- `tests/test_spsc.py` (9): priming gate, FIFO exactness across wraparound,
  underrun zero-fill + re-prime, backpressure blocking, push timeout/stop abort,
  pop > capacity clamp, validation, threaded producer/consumer order preservation.
- `tests/test_audio_worker.py` (5): worker stream == one-shot frozen-DDC
  reference (sample-exact); band change applies between blocks and keeps
  streaming; refused band (M2a Nyquist-wrap guard) keeps old band + stream
  alive; lapped-reader gap counted, stream survives (policy (a)); volume is
  plain attenuation.
- `tests/test_m2b_sim_pipeline.py` (1): realtime end-to-end Sim acceptance —
  0 underruns, played stream sample-exact vs reference, 5 kHz peak, 0 in-Xrun.

## Honest notes / accepted limits
- **Human listening check remains for Kali**: the machine verified the 5.4 kHz
  line numerically in the *played* stream and on real speakers, but "ears on
  the drag-to-listen workflow" (and how objectionable the policy-(a) clicks
  feel) is a human judgment.
- **End-of-stream priming gate**: if the producer stops while the ring is
  unprimed, a tail < prebuffer stays gated. Irrelevant for continuous live
  monitoring; documented in `spsc.py` and pinned by test.
- `push()` partial-write on timeout/stop leaves a truncated-but-ordered stream
  (worker counts it as `n_push_failed`); only reachable at teardown.
- Output latency ≈ prebuffer (80 ms default) + worker poll (10 ms) + capture
  block (1 ms) — a monitoring-grade choice, tunable via `--prebuffer-ms`.
- `--volume` is a fixed attenuator for speaker safety. AGC / normalize /
  compressor remain M3 (`gain.py` still a stub).

## Freeze status after M2b
- `ultrascan/dsp/spec_audio.py`, `ultrascan/dsp/audifier.py` — FROZEN, diff empty this milestone.
- No new freezes: the M2b modules are plumbing, not contract surfaces.
- Still stubs: `gain.py`→M3, `detect/detector.py`+`events.py`→M4, `record/guano_writer.py`→M5.
