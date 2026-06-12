# M2a — DDC Numerical Correctness Report (offline, no audio output)

## Scope
First half of M2 (DESIGN §6 M2 / §5): the image-free heterodyne DDC chain as a
block-streaming `HeterodyneAudifier` (frozen §4.2 `Audifier` protocol), verified
**numerically offline**. Output callback / SPSC queue / speakers / GUI band
selection are M2b and were NOT touched. `gain.py` / detect / record remain stubs.

## Implementation (`ultrascan/dsp/audifier.py`, FROZEN at M2a)
Chain per `process(block)` call (input real @ fs_in, output real float32 @ 48 kHz):
1. complex NCO `exp(-j2π·f_lo_sel·n/fs)` — phase accumulated across blocks (mod 2π)
2. mix — selected band lower edge lands at DC
3. **one-sided complex FIR LPF**, passband [0, cutoff], cutoff = min(W, fs_dec/2),
   255 taps, gain 2 — `lfilter` zi carried across blocks
4. ÷5 integer decimation (250k → 50k) — pick phase carried across blocks
5. realify
6. soxr **streaming** resample 50k → 48k — resampler state carried across blocks

Plus `flush()` (offline/one-shot only): drains the resampler tail and ends the
stream — live streaming never calls it. `configure()` is **atomic**: it validates
everything before mutating, so a refused re-selection leaves a running stream intact.

### Why the LPF is complex one-sided (the 解析信号 of DESIGN §1)
With real-coefficient taps, `real()` commutes with the filter, so
`real(LPF(s·e^{-jωn})) ≡ LPF(s·cos(ωn))` — mathematically identical to the
**prohibited** real-cos mix, and a tone at the image frequency `f_lo−Δ` folds
onto `+Δ`. The one-sided passband keeps only positive baseband frequencies, so
the image is rejected *before* realify. This is what makes the §9 mirror test
pass at all; it is the substance of 「複素ミックス（解析信号）＝鏡像なし」.

### Configure-time validity guards (from the pre-freeze adversarial review)
- **`f_lo_sel + cutoff <= fs_in/2`** — beyond it, the mixed-down negative line of
  a real tone wraps mod fs to `fs − f − f_lo` INSIDE the passband and returns at
  full gain (the §11 defect through the back door). `configure()` raises. **(fatal finding)**
- **`bandwidth >= one FIR transition width`** (≈3.24 kHz at 250k / 255 taps) —
  narrower selections would pass nothing but filter skirt. `configure()` raises.

`ddc.py` is now a thin offline one-shot wrapper around the audifier (used by
tests/analysis); it flushes so the buffer tail is not dropped.

## Measured results (250 kHz input, W = 10 kHz)
| Check | Result |
|---|---|
| Mapping: 45 kHz tone, LO = 40 kHz | output peak **5000.2 Hz** |
| Mirror: 35 kHz tone (= LO − 5 kHz), complex mix | **−61 dB** vs desired response |
| Mirror: same tone through real-cos reference chain | **0 dB** (image as loud as desired — the defect §11 prohibits) |
| Continuity: whole vs 7 uneven chunks (1…88500 samples) | identical length, max diff **1.2e-7** |
| Passband amplitude (gain-2 taps): 0.5-amp tone | output RMS **0.35355** = ideal 0.5/√2 (no analytic halving) |
| One-shot flush: 1.0 s @ 250 kHz | body 47547 + tail **453** = **48000** samples exactly (= 250k ÷5 ×48/50) |
| Real WAV (`captures/m0_ultramic_keys_250k.wav`), LO = 20 kHz | peak **5430.2 Hz** = M0's offline-FFT repeller (25.43 kHz) − 20 kHz, 78% of energy in 3–11 kHz |

## Tests
`pytest -q`: **50 passed** — 33 pre-existing untouched + 17 new in
`tests/test_audifier.py`:
- frequency mapping (45k/LO40k → 5k; 25k/LO20k → 5k)
- mirror test incl. **contrast against a test-local prohibited cos-mix reference**
  (image suppressed ≥ 50 dB by the complex mix, present at full strength in cos mix)
- continuity: chunked == single-call (NCO phase / FIR zi / decim phase / soxr state)
- real-capture WAV repeller maps to 5.2–5.7 kHz (skips if the gitignored WAV is absent)
- protocol conformance, configure validation, empty-block, reconfigure-resets-state
- configure guards: wrap-into-passband (the fatal review finding, incl. the
  cutoff-clamp case and the legal `f_lo + cutoff == fs/2` boundary),
  sub-transition bandwidth, refused-reconfigure atomicity
- passband amplitude (gain-2 taps), RMS pinned to 0.5/√2 ± 0.01
- one-shot flush: tail recovery + exact 48000-sample length; flush ends the
  stream (process raises) until reconfigured

## Notes / accepted limits
- `configure()` mid-stream fully resets state (band re-selection is not click-free) —
  acceptable for v1, noted in the docstring.
- soxr streaming holds ~453 samples of latency; output for 1 s of input is ~47.5k
  samples. The tail stays in the resampler — recovered by `flush()` in offline /
  one-shot use (`ddc_heterodyne`); live streaming never flushes.
- (superseded) An earlier draft treated the wrap-around self-image as "outside any
  practical band selection". The adversarial review upgraded exactly that to the
  fatal finding: nothing stopped such a selection, so `configure()` now refuses
  `f_lo + cutoff > fs/2` instead of relying on operator discipline.

## Pre-freeze adversarial review — all 7 confirmed findings closed

| # | Finding | Disposition |
|---|---|---|
| 1 | **(fatal)** `f_lo + cutoff > fs/2`: wrapped image `fs−f−f_lo` re-enters the passband at full gain | **Fixed** — `configure()` guard; pinned by `test_configure_rejects_selection_wrapping_into_passband` |
| 2 | Real-coefficient LPF lets `real()` commute with the filter ≡ the PROHIBITED cos mix (image folds back) | **Fixed** — one-sided complex FIR; pinned by both mirror tests incl. the cos-mix contrast |
| 3 | Analytic-signal step halves passband amplitude | **Fixed** — gain-2 taps; pinned by `test_passband_amplitude_survives_analytic_halving` (RMS 0.35355 = ideal) |
| 4 | One-shot/offline use silently dropped the soxr tail (~453 samples) | **Fixed** — `flush()` added, `ddc_heterodyne` flushes; pinned by `test_oneshot_flush_recovers_resampler_tail` |
| 5 | `bandwidth` below one FIR transition width selects nothing but skirt | **Fixed** — `configure()` guard; pinned by `test_configure_rejects_bandwidth_narrower_than_fir_transition` |
| 6 | Finite 255-tap skirt: image rejection reaches full depth only ≳½ transition (~1.6 kHz) below LO; band edges roll off | **Rejected with reason** — physics of any finite FIR; documented in the class docstring; ≥50 dB at 5 kHz spacing pinned by the mirror test |
| 7 | `configure()` re-selection fully resets stream state — band switch is not click-free | **Rejected with reason** — accepted for v1 (a crossfade is an M2b+ concern); documented; reset semantics pinned by `test_reconfigure_resets_stream_state` |

Hardening found during close-out: `configure()` used to mutate attributes before
the wrap guard ran, so a refused mid-stream re-selection left stale/lying
attributes behind. Now atomic (validate-then-commit); pinned by
`test_failed_reconfigure_leaves_stream_intact`.

## Freeze status after M2a
- `ultrascan/dsp/audifier.py` — **FROZEN** (joins `spec_audio.py`).
- Still stubs: `gain.py`→M3, `detect/detector.py`+`events.py`→M4, `record/guano_writer.py`→M5.
