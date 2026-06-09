# M0 — Capture Stress Test Report

## Source
- **name**: wasapi
- **is_synthetic**: False
- **samplerate**: 250000.0
- **blocksize**: 2048
- **channels**: 1
- **device**: 23
- **mode**: WASAPI-exclusive

## WASAPI exclusive native-rate check
- **requested_rate**: 250000.0
- **ok**: True
- **error**: None
- **device_name**: マイク (2- UltraMic 250K 16 bit r4)
- **device_default_rate**: 48000.0

## CAPTURE FAILED (real-HW blocker)
- error: `PortAudioError: Error opening InputStream: Invalid device [PaErrorCode -9996]`
- No native-rate stream opened, so the >24 kHz energy verdict could **not** be measured (see FFT section).
- Note: `IsFormatSupported` (the native-rate check above) passing while the stream open fails is typically a WASAPI **exclusive** `Initialize` refusal — device held by another app, or exclusive control disabled.

## Capture
- requested duration: 10.00 s
- elapsed: 0.01 s
- samplerate: 250000 Hz (Nyquist 125000 Hz)
- blocksize: 2048 samples  (block period 8.19 ms)
- callbacks: 0
- total frames: 0

## Xrun / overrun
- input_overflow (Xrun) count: **0**
- input_underflow count: 0
- dummy load injected: 0.0 ms/callback
- callback service time (ms): min 0.00 / mean 0.00 / p99 0.00 / max 0.00

## FFT verification (offline, after capture)
- ultrasonic energy (>24 kHz): **NOT MEASURED** — no samples were captured (capture did not open).

## Notes
- CAPTURE FAILED before/while streaming: PortAudioError: Error opening InputStream: Invalid device [PaErrorCode -9996]

## Overload boundary (dummy-load test, DESIGN §6 M0)
- reproduce with `--load-ms <ms>`; a per-callback load above the 8.19 ms block period flags every block as an overrun. Observed Sim behaviour: graceful degradation (overruns flagged, no crash).
- host note: Windows `time.sleep` granularity is ~15 ms, so Sim pacing is coarse — overrun detection is keyed to consumer-callback duration, not host timer jitter. Real per-block latency/Xrun limits are an HW measurement (Kali).

## What M0 establishes / hands off to Kali
- **Kali, on real hardware, confirm:**
  1. `--source wasapi --duration 10` opens 250k exclusive (native-rate check OK).
  2. real energy appears above 24 kHz (e.g. jingling keys / ultrasonic remote).
  3. sweep `--blocksize` down; find the smallest with 0 Xrun over several minutes.
  4. `--load-ms` past the block period only stalls audio (no crash).

---

## Real-HW investigation log — 2026-06-09 (device: UltraMic 250K 16 bit r4, WASAPI idx 23)

> Status: **live capture blocked**, so the >24 kHz energy verdict is **NOT MEASURED** above.
> The success version of this report replaces this file once a native-rate stream opens.
> This log is the investigation record; the auto-generated sections above are the tool output.

### Confirmed (independent of being able to stream)
1. **Native format = 250 kHz, mono, 16-bit.** WASAPI `IsFormatSupported` (exclusive) accepts
   only `250000 Hz / ch=1`; it rejects 192k / 256k / 384k and stereo (`Invalid sample rate`).
   So the device's true native capture format is mono 250k — established via the format query
   even though the stream itself won't open.
2. **Shared-mode 48k resample trap is correctly rejected.** Opening shared (non-exclusive) at
   250k fails `Invalid sample rate` (shared offers only the ~48k mix rate). The code never falls
   back to shared, so it cannot silently resample and destroy >24 kHz content (DESIGN §2/§11).
3. **Xrun count = PortAudio's real flag**, not the Sim model. The WASAPI callback reads
   `CallbackFlags.input_overflow` (`ultrascan/capture/sources.py:334`). The Sim heuristic
   (`prev_dt > period`) lives only in `_ThreadedSource._run`, which `WasapiExclusiveSource`
   does not inherit.

### The blocker
- WASAPI **exclusive** open fails `Invalid device [PaErrorCode -9996]`, **100% consistently**:
  raw sounddevice (bypassing our code), callback **and** blocking read, blocksize
  {0,256,480,512,750,1024,1250,2048,2500,4096}, dtype {int16, float32}, latency {low, high,
  2–20 ms}. `IsFormatSupported` passes but `IAudioClient::Initialize` fails → an exclusive-grab
  refusal, not a format/code issue (our code would stream the moment the device opens).
- **No WDM-KS fallback:** the UltraMic enumerates only under MME / DirectSound / WASAPI.
  MME @250k → host error; DirectSound @250k → `E_INVALIDARG`. WASAPI-exclusive is the only
  native path, and it's the one refusing `Initialize`.
- Registry: **two** UltraMic capture endpoints — active (`DeviceState=1`, `{f17a74df…}`) and a
  stale **not-present** (`DeviceState=4`, `{37d66efa…}`). "Allow exclusive control" flag is at
  default (= allowed) for both, so the Windows checkbox is probably not the cause.
- Stack: PortAudio `V19.7.0-devel`, sounddevice `0.5.5`, numpy `1.26.4`, Python `3.9.13`.

### Isolation plan (Python/PortAudio-specific vs device/OS-level)
- **A. Pipeline half, independent of live capture:** record a 250k WAV with SeaWave or
  Audacity (WASAPI, 250000 Hz), then `--source wav --wav <file>` runs it through the *same*
  `_fft_check`, producing the real >24 kHz energy %. Confirms the DSP/FFT path without capture.
- **B. Live capture:** release the device + reconnect the UltraMic (index changes →
  `--list-devices`), then `--source wasapi --device <idx>`. If it opens, sweep `--blocksize`
  down to the Xrun-zero floor over several minutes and regenerate this report as the success
  version.
- **C. If SeaWave/Audacity capture 250k but Python WASAPI alone still returns -9996** →
  stack-specific confirmed. Before ASIO, try in order: (a) a different sounddevice / PortAudio
  build, (b) a different backend (e.g. the `soundcard` library).
- **ASIO is on hold (by decision):** stock sounddevice/PortAudio ships without ASIO (Steinberg
  licensing), and the UltraMic is a class-compliant USB device with no vendor ASIO driver — so
  ASIO would mean an ASIO4ALL detour. Not implemented now.
