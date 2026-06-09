# M0 — Capture Stress Test Report

> **SYNTHETIC-ONLY** — generated from a Sim source (no microphone).
> Real 250k native capture, Xrun limits and audibility are judged on
> real hardware by **Kali**. These numbers validate the *pipeline*, not the HW.

## Source
- **name**: synthetic
- **is_synthetic**: True
- **samplerate**: 250000.0
- **blocksize**: 2048
- **channels**: 1
- **kind**: tone
- **tone_hz**: 45000.0
- **amplitude**: 0.5

## Capture
- requested duration: 3.00 s
- elapsed: 3.03 s
- samplerate: 250000 Hz (Nyquist 125000 Hz)
- blocksize: 2048 samples  (block period 8.19 ms)
- callbacks: 369
- total frames: 755712

## Xrun / overrun
- input_overflow (Xrun) count: **0**
- input_underflow count: 0
- dummy load injected: 0.0 ms/callback
- callback service time (ms): min 0.00 / mean 0.03 / p99 0.09 / max 0.27

## FFT verification (offline, after capture)
- nfft: 65536
- peak frequency: **45.00 kHz**
- energy fraction above 24 kHz: **100.00%**
- ultrasonic energy present (>24 kHz): **YES**

## Notes
- SYNTHETIC-ONLY: Sim source. Real 250k native capture, Xrun limits and audibility are judged on real hardware by Kali.

## Overload boundary (dummy-load test, DESIGN §6 M0)
- reproduce with `--load-ms <ms>`; a per-callback load above the 8.19 ms block period flags every block as an overrun. Observed Sim behaviour: graceful degradation (overruns flagged, no crash).
- host note: Windows `time.sleep` granularity is ~15 ms, so Sim pacing is coarse — overrun detection is keyed to consumer-callback duration, not host timer jitter. Real per-block latency/Xrun limits are an HW measurement (Kali).

## What M0 establishes / hands off to Kali
- **establishes (Sim):** the capture pipe runs end-to-end — InputSource -> copy-only callback -> ring -> offline FFT — and that a known >24 kHz tone is recovered, overruns are detected, and overload degrades gracefully.
- **does NOT establish:** that this PC + UltraMic can actually open WASAPI **exclusive** at 250k, nor the real Xrun-free blocksize floor.
- **Kali, on real hardware, confirm:**
  1. `--source wasapi --duration 10` opens 250k exclusive (native-rate check OK).
  2. real energy appears above 24 kHz (e.g. jingling keys / ultrasonic remote).
  3. sweep `--blocksize` down; find the smallest with 0 Xrun over several minutes.
  4. `--load-ms` past the block period only stalls audio (no crash).
