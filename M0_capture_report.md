# M0 — Capture Stress Test Report

## Source
- **name**: wasapi
- **is_synthetic**: False
- **samplerate**: 250000.0
- **blocksize**: 256
- **channels**: 1
- **device**: 23
- **mode**: WASAPI-exclusive

## WASAPI exclusive native-rate check
- **requested_rate**: 250000.0
- **ok**: True
- **error**: None
- **device_name**: マイク (2- UltraMic 250K 16 bit r4)
- **device_default_rate**: 48000.0

## Capture
- requested duration: 12.00 s
- elapsed: 12.02 s
- samplerate: 250000 Hz (Nyquist 125000 Hz)
- blocksize: 256 samples  (block period 1.02 ms)
- callbacks: 11685
- total frames: 2991360

## Xrun / overrun
- input_overflow (Xrun) count: **0**
- input_underflow count: 0
- dummy load injected: 0.0 ms/callback
- callback service time (ms): min 0.00 / mean 0.01 / p99 0.03 / max 0.16

## FFT verification (offline, after capture)
- nfft: 65536
- peak frequency (overall): **25.43 kHz**
- peak frequency (>24 kHz band): **25.43 kHz**
- energy fraction above 24 kHz: **33.15%**
- ultrasonic energy present (>24 kHz): **YES**

## Blocksize Xrun sweep + multi-minute soak (real HW, manual summary)
Descending `--blocksize` ladder on device 23 (UltraMic 250K, WASAPI-exclusive, 250 kHz).
The pest repeller (a constant, frequency-sweeping ultrasonic source in the room) supplied
the >24 kHz signal, so every run also reported `ultrasonic=YES`.

| blocksize | block period | duration | callbacks | Xrun |
|----------:|-------------:|---------:|----------:|-----:|
|      2048 |     8.192 ms |     10 s |     1 211 |  **0** |
|      1024 |     4.096 ms |     20 s |     4 843 |  **0** |
|       512 |     2.048 ms |     20 s |     9 746 |  **0** |
|       256 |     1.024 ms |     20 s |    19 511 |  **0** |
|       128 |     0.512 ms |     20 s |    38 991 |  **0** |
|        64 |     0.256 ms |     20 s |    78 040 |  **0** |
|        32 |     0.128 ms |     15 s |   116 922 |  **0** |
|        16 |     0.064 ms |     15 s |   233 844 |  **0** |
|         8 |     0.032 ms |     10 s |   311 688 |  **0** |
|         4 |     0.016 ms |     10 s |   622 752 |  **0** |

**Soak:** blocksize 256 for **180 s** → 175 751 callbacks, 44 992 256 frames, **Xrun 0**.

**Finding:** no Xrun-limited floor was reached — 0 Xrun held all the way down to blocksize 4.
On this machine the requested blocksize is just the PortAudio callback granularity; WASAPI
exclusive manages its own hardware period underneath, so buffer size is **not** the
Xrun-limiting factor here. Blocksize is therefore a latency/robustness choice, not a
correctness one. **Recommended default: 256** (≈1.0 ms latency) — soaked 3 min clean above;
raise toward 1024–2048 for extra scheduling headroom on long unattended sessions.

**Regression asset:** `captures/m0_ultramic_keys_250k.wav` — 12 s @ 250 kHz float32
(key-jingle + repeller), replayable through the same pipe via `--source wav`. Gitignored.

## Overload boundary (dummy-load test, DESIGN §6 M0)
- reproduce with `--load-ms <ms>`; a per-callback load above the 1.02 ms block period flags every block as an overrun. Observed Sim behaviour: graceful degradation (overruns flagged, no crash).
- host note: Windows `time.sleep` granularity is ~15 ms, so Sim pacing is coarse — overrun detection is keyed to consumer-callback duration, not host timer jitter. Real per-block latency/Xrun limits are an HW measurement (Kali).

## What M0 establishes (real hardware — CONFIRMED)
- **1. 250k WASAPI-exclusive opens** — native-rate check OK; stream ran at 250000 Hz (NOT resampled to the 48000.0 Hz share-mode mix rate).
- **2. real >24 kHz energy captured** — verdict YES; 33.15% of power above 24 kHz, ultrasonic-band peak 25.43 kHz.
- **3. Xrun at blocksize 256** — 0 overruns over 12.0 s (block period 1.02 ms). Sweep `--blocksize` down to find the Xrun-free floor.
- **4. overload boundary** — `--load-ms` past the block period only stalls audio (graceful, no crash).
