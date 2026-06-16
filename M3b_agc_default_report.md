# M3b — AGC default-ceiling tuning (post-M3, pre-M4)

> **Not a new milestone.** A single human-authorized default-value change before
> M4, to make the M3 AGC usable on real HW. Sim-verified here; the "sounds right"
> judgment is **Kali's real-HW ear check** (DESIGN §6 roles). **Held uncommitted**
> pending that check — the value is trivial to retune.

## Problem (real-HW finding)
The M3 `AGCGain` default `max_gain=100` (+40 dB) is too high in practice. On a
quiet band or a weak signal the AGC climbs to the ceiling and drives the **noise
floor and low-frequency content all the way to the `target_rms`** — a roar that
buries even speech. This is the *designed* "gain raises the noise floor" behavior
(DESIGN §6 M3), but the **default ceiling** was not practical. Without `--agc`
(plain M2b) the band is clear, so the DDC has no low-frequency leak — the cause
was isolated to the AGC ceiling alone.

## Change (minimal — one lever)
| Param | Old default | New default | Note |
|---|---:|---:|---|
| `max_gain` | 100.0 (+40 dB) | **12.0 (≈ +21.6 dB)** | the fix |
| `target_rms` | 0.2 | 0.2 (**unchanged**) | comfortable level; not the cause |
| `attack_s` / `release_s` | 0.010 / 0.300 | unchanged | time constants, not the cause |

Capping at **12** bounds the worst-case lift to **12×** instead of 100× — an
**8.3× (≈ 18.4 dB)** reduction in the maximum floor-lift — while a merely *quiet*
signal (in-band RMS ≳ `target_rms/max_gain` ≈ 0.017) still reaches the target.
Only `max_gain` is touched (minimal change, per the work order). Manual override
is preserved: `--agc-max-gain 100` restores the old behavior; any value can be
dialed while listening.

Touched: `ultrascan/dsp/gain.py` (default literal + docstrings + a freeze-scope
note), `scripts/m2b_listen.py` (`--agc-max-gain` default 100 → 12 + help text).

## Freeze discipline (DESIGN §4 / §11)
The work order explicitly authorized this: it is a **default-value** change, not a
**signature** change — the freeze is on the `GainStage.process(audio) -> audio`
contract, which is byte-for-byte unchanged.
- `gain.py` diff = only `max_gain: float = 100.0` → `= 12.0` plus docstring text.
  `AGCGain.__init__` parameter list / kinds / annotations and `process()` are
  **unchanged**.
- `ultrascan/dsp/spec_audio.py`, `ultrascan/dsp/audifier.py` — `git diff` **empty**.

## Tests — `tests/test_gain.py` (now 78 passed, 1 skipped)
The 1 skip is unchanged (gitignored real-capture WAV). Re-pinned to the new
*correct* behavior (not loosened to pass):
- **`test_quiet_signal_is_lifted_toward_target`** — input 0.01 → **0.03** (a
  weak-but-*reachable* tone, in_rms 0.021): still climbs to the target (tail RMS
  ~0.20, gain ~9.4). The old 0.01 input is below the reachable level under a 12×
  ceiling, so it now belongs to the bounded case below.
- **`test_faint_signal_bounded_by_conservative_default`** (NEW — the regression
  guard for this fix): a noise-floor-faint input (0.005, in_rms 0.0035) is lifted
  only to **~ceiling× input ≈ 0.042** under the default ceiling, vs **~0.20**
  under the old `max_gain=100`. Pins "no longer slams faint signal/noise to the
  target."
- **`test_noise_floor_rises_with_gain`** — **kept**. The floor still rises with
  gain (tail 12× input > the asserted 10×); the documented physical limit stands,
  now bounded by the ceiling.
- `test_attack_is_faster_than_release` — stale ceiling literal `100.0` → `12.0`
  (cosmetic; `min(0.2/0.02, ·) = 10` either way, no behavior change).

## Sim verification (real CLI audio path, wall-clock 48 kHz; `--source synthetic --sim-out`)
45 kHz tone, LO 40 kHz → 5 kHz audible, `--agc` (defaults: target 0.2, ceiling 12):

| Run | tone-amp | settled gain | played RMS | peak | q_underruns | max_zero_run |
|---|---:|---:|---:|---:|---:|---:|
| Weak, reachable | 0.05 | 5.65× | **0.192** (≈ target) | 5000 Hz | **0** | 0 |
| Faint (M3's roar input) | 0.01 | **12.0× (capped)** | **0.081** (bounded) | 5000 Hz | **0** | 0 |
| Weak, **15 s soak** | 0.05 | 5.65× | 0.197 | 5000 Hz | **0** | 0 |

- **Reaches target for quiet:** amp 0.05 → RMS 0.19–0.20 at gain 5.65. ✔
- **Bounded for faint (the fix):** the *same* faint input M3 drove to RMS **0.188**
  (gain 28×, ceiling 100) is now bounded to **0.081** at gain 12. ✔
- **Continuity:** 0 queue underruns / 0 input xruns / 0 dropped / max_zero_run 0
  in every run incl. the 15 s soak; gain stable (no pumping). ✔

## Status — STOPPING here (work-order Step 1 complete)
Awaiting **Kali's real-HW ear check**: with `--agc` and the default ceiling,
confirm it is *not* a roar and a weak ultrasonic source is easier to hear. **Not
committed** — held in the working tree so the ceiling can be retuned (8–16 was the
suggested range; 12 is the middle) after listening, before the M4 go-ahead.
**M4 (detection/measurement) not started.**
