# M3 — Gain / AGC Report (small-signal amplification)

## Scope
DESIGN §4.3 / §6 M3: a `GainStage` inserted as **one independent stage** between
the frozen DDC output and the SPSC queue, implementing `AGCGain` (continuous-mode
default). Audio side only — display contrast (M1) untouched. Detection/recording
remain stubs (M4/M5). Frozen files untouched (`spec_audio.py`, `audifier.py` —
`git diff` empty).

## What M3 implements (and what it does NOT)
- `GainStage` Protocol **confirmed and frozen** — unchanged single-method
  signature `process(audio) -> audio`, deliberately sized for all THREE planned
  implementations (see below). `gain.py` is now FROZEN (joins `spec_audio.py`,
  `audifier.py`).
- `AGCGain` is the **only** implementation built this milestone. `NormalizeGain`
  (snapshot) and `CompressorGain` are later milestones, added as NEW classes —
  the frozen file is not re-edited for them (same discipline as `audifier.py` at
  M2a, where only `HeterodyneAudifier` landed).

### Why one frozen `process(audio) -> audio` is enough for all three
| Impl | Mode | Fits the contract because |
|---|---|---|
| `AGCGain` (M3) | continuous, stateful | block in → level-adjusted block out; current gain carried across blocks |
| `NormalizeGain` (later) | snapshot | `process()` called once over the whole snapshot array → scale to a fixed peak/RMS |
| `CompressorGain` (later) | continuous, stateful | per-block dynamic-range compression |

All three are "audio block in → level-adjusted block out", so **no signature
change / human approval is needed** — the M3 freeze stands on the existing shape.

## `AGCGain` algorithm (`ultrascan/dsp/gain.py`)
Per `process(block)` call (audio is float @ 48 kHz, the audifier output rate):
1. `level = block RMS` (floored by `eps=1e-6`, a divide-by-zero guard — **NOT** a
   noise gate).
2. `g_want = clip(target_rms / level, 0, max_gain)` — boosts quiet, attenuates loud.
3. One-pole the gain toward `g_want` with an **asymmetric** time constant:
   - gain must **drop** (signal got loud) → `attack_s` (fast: kills overshoot
     before it reaches the speaker),
   - gain must **rise** (signal got quiet) → `release_s` (slow: avoids "pumping").
   `alpha = 1 − exp(−(n/fs)/tau)` uses the **actual** block length `n`, so the
   time constants stay correct under the worker's variable block sizes.
4. Apply `linspace(g_prev, g_new, n)` — a **per-sample linear ramp** within the
   block. Combined with `g_new(block j) == g_start(block j+1)`, the gain curve is
   **continuous across block boundaries → no click** (DESIGN §5 continuity rule).

State carried across blocks: the current gain (`_gain`). `reset()` (AGC-specific,
not on the Protocol) drops it to unity, e.g. on a band re-selection.

Defaults: `target_rms=0.2`, `attack_s=0.010`, `release_s=0.300`, `max_gain=100`
(+40 dB), exposed on the CLI as `--agc --agc-target --agc-attack-ms
--agc-release-ms --agc-max-gain`.

## Audio-path insertion (`ultrascan/audio/worker.py`)
`AudioWorker` gained an optional `gain=` parameter. In the worker loop:
`samples → audifier.process() → gain.process() → (volume) → SPSC.push()`.
- An **independent stage** (DESIGN §3): the frozen audifier is unaware of it.
- `gain=None` (default) is identity → the **M2b path is byte-for-byte unchanged**;
  all pre-existing worker tests construct the worker without `gain` and pass
  untouched.
- Driven on the **worker thread only** → AGC state never races; never touched in
  the output callback (DESIGN §2 — copy/drain only in callbacks).
- `--volume` remains a separate fixed safety attenuator applied after the gain.

## Constraint stated explicitly (DESIGN §6 M3) — in code, CLI, and here
**Raising gain raises the noise floor with the signal.** AGC helps a signal that
is merely *quiet*; a signal buried *under* noise is **not** separated by plain
gain (spectral subtraction / noise gate is a future phase). The mic self-noise /
ADC noise floor is the physical bottom — during silence the AGC climbs to
`max_gain` and lifts that floor toward the target. This is documented in the
`gain.py` module/class docstrings, the `--agc-max-gain` CLI help, the CLI banner,
and pinned by `test_noise_floor_rises_with_gain`. **Display gain (waterfall
contrast) and audio gain (AGC volume) are separate knobs** — M3 touches only the
audio side.

## Verification — Sim first (numeric; no hardware)
The real audio path (`m2b_listen.py --source synthetic --sim-out`, wall-clock
48 kHz drain), faint/loud 45 kHz tone, LO 40 kHz → 5 kHz audible:

| Run | played-body RMS | settled gain | peak | queue underruns | max_zero_run |
|---|---:|---:|---:|---:|---:|
| Faint tone amp 0.01, **AGC OFF** | **0.0071** | — | 4999.9 Hz | **0** | 0 |
| Faint tone amp 0.01, **AGC ON** (target 0.2) | **0.1883** | 28.27× | 5000.0 Hz | **0** | 0 |
| Loud tone amp 0.9, **AGC ON** (target 0.2) | **0.2019** | 0.314× | 4999.9 Hz | **0** | 0 |
| Faint tone, AGC ON, **30 s soak** | **0.1984** | 28.26× | 5000.0 Hz | **0** | 0 |

- **Lift confirmed numerically**: faint signal RMS 0.0071 → 0.1883 (~26× lift),
  settling near the 0.2 target. **OFF vs ON difference is the headline number.**
- **Attenuation confirmed**: a loud tone is pulled DOWN (gain 0.314×) to RMS 0.202
  — AGC steers from both directions toward the target.
- **AGC is a level stage, not a frequency stage**: peak stays at 5000 Hz in every
  run.
- **Continuity / underruns**: inserting AGC keeps **queue underruns 0** and
  **max_zero_run 0** in every run, incl. a 30 s soak (`q_popped_zero=4067` = the
  start-of-stream priming silence, exactly as in M2b; underruns 0 → no mid-stream
  gap). The per-sample gain ramp carries gain state across blocks → no boundary
  click.

### Unit tests — `tests/test_gain.py` (13 new)
`pytest -q`: **77 passed, 1 skipped** — the 64 pre-existing tests **untouched**
(the 1 skip is the gitignored real-capture WAV, absent on this machine), 13 new:
- protocol conformance (`AGCGain` is a `GainStage`), constructor validation,
  empty-block passthrough;
- quiet signal lifted toward target / loud signal attenuated toward target /
  steady signal converges to target;
- **continuity**: per-sample gain reconstructed from a constant input has **no
  boundary discontinuity** (each boundary step ≤ the surrounding ramp slope) — a
  click (state not carried) would be order-unity; state-persistence across blocks
  (`g_end(block N) == g_start(block N+1)`);
- **attack ≪ release**: in one equal-length block the fractional move toward the
  wanted gain is larger when reducing gain (attack) than raising it (release) —
  the overshoot-vs-pumping trade-off;
- silence gain capped at `max_gain` (no blow-up); **noise floor rises with gain**
  (the documented limit); `reset()` returns gain to unity;
- **worker integration**: `DDC → AGC → SPSC` lifts a faint band (~6×+ RMS) while
  the peak stays at 5 kHz.

## Honest notes / accepted limits
- **SYNTHETIC-ONLY here**: all numbers above are Sim. `captures\` does not exist
  on this machine, so the saved real-capture WAV (`m0_ultramic_keys_250k.wav`)
  and live UltraMic verification (faint ultrasonic source, e.g. a distant key
  jingle, AGC on; underrun re-measurement) are **Kali's real-HW judgment**
  (DESIGN §6 roles). The DSP is fully pinned numerically; "does it sound easier to
  hear" is a human call.
- **venv is Python 3.11.9**, not the 3.9.13 named in the work order. v1 has no
  torch (DESIGN "Python の判断" option 2), so this is not a blocker; flagged for
  when v2 wires in sigscan's CNN.
- **Band change does not auto-reset AGC**: after a re-selection the AGC re-adapts
  via attack/release rather than snapping. `reset()` exists if a caller prefers a
  hard reset; the worker does not call it (re-adaptation is acceptable for v1).
- AGC level detection is **block-RMS based** with a per-sample gain ramp — not a
  per-sample envelope follower. Continuous and fast (fully vectorized, runs on the
  worker thread, never the callback); adequate for monitoring. A sample-accurate
  envelope follower is out of scope for M3's minimal implementation.

## Freeze status after M3
- `ultrascan/dsp/spec_audio.py`, `ultrascan/dsp/audifier.py` — FROZEN, diff empty.
- **`ultrascan/dsp/gain.py` — NEWLY FROZEN at M3** (`GainStage` Protocol + `AGCGain`).
- Still stubs: `detect/detector.py` + `events.py` → M4, `record/guano_writer.py` → M5
  (diff empty this milestone).

---

> **Post-M3 update (pre-M4):** the `AGCGain` `max_gain` **default** was lowered
> **100 → 12** (+40 dB → +21.6 dB) after the real-HW "roar" finding — the +40 dB
> ceiling drove a quiet band / noise floor to the full target. Signature
> unchanged (human-approved default tune; see the `gain.py` freeze note). Details
> and new Sim numbers: **`M3b_agc_default_report.md`**. The faint-tone RMS 0.188
> in the table above was measured at the original ceiling 100.
