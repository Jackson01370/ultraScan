"""FROZEN CONTRACT §4.3 — GainStage (small-signal amplification).

╔══════════════════════════════════════════════════════════════════════════╗
║  FROZEN at M3.  From here on `git diff --stat` for this file must stay     ║
║  empty; signature changes need human approval (DESIGN §4, §11).            ║
╚══════════════════════════════════════════════════════════════════════════╝

NOTE (post-M3, human-approved): the freeze is on the SIGNATURE -- the
``GainStage.process(audio) -> audio`` contract and the constructor parameter
list -- NOT on default VALUES. Tuning a default value with approval is allowed;
done once here: ``max_gain`` 100 -> 12, after the real-HW finding that the
+40 dB ceiling lifted a quiet band / noise floor to a roar (CLI
``--agc-max-gain`` still overrides per-run).

The ``GainStage`` Protocol is fixed here looking ahead to all THREE planned
implementations, so the single-method contract is enough for every one of them:

  * ``AGCGain``       — continuous-mode default (M3, implemented below). Tracks
                        block level and steers a *stateful* gain toward a target;
                        state (current gain) persists across blocks.
  * ``NormalizeGain`` — snapshot mode (later M). ``process()`` is called once over
                        a whole snapshot array → scale to a fixed peak/RMS.
  * ``CompressorGain``— dynamic-range compression (later M). Per-block, stateful.

All three are "audio block in → level-adjusted audio block out", so the frozen
contract is exactly ``process(audio) -> audio`` — no extra methods needed. Later
implementations are NEW classes (new or this module); this signature is not
edited for them (same discipline as Audifier at M2a).

Reminder (DESIGN §4.3): display gain (waterfall contrast) and audio gain (volume)
are SEPARATE knobs — never conflate them. This module is the AUDIO side only;
M1 display contrast is untouched.

Physical limit (DESIGN §6 M3): raising gain raises the NOISE FLOOR with the
signal. AGC helps a signal that is merely *quiet*; a signal buried *under* noise
is NOT separated by plain gain (spectral subtraction / noise gate is a future
phase). The mic self-noise / ADC noise floor is the physical bottom — AGC will
happily lift that floor toward the target during silence.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class GainStage(Protocol):
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Audio block in -> level-adjusted audio block out.

        Stateful implementations (AGC, compressor) MUST carry their state across
        blocks: a gain discontinuity at a block boundary is an audible click
        (same continuity rule as the Audifier, DESIGN §5)."""
        ...


class AGCGain:
    """Automatic gain control — keep the output level near a target (DESIGN §4.3).

    Block-streaming and stateful: the current gain persists across ``process()``
    calls so the level is steered smoothly over the whole live stream, never
    reset per block. Within each block the gain is applied as a per-sample linear
    ramp from the previous gain to the new gain, so there is **no discontinuity
    at a block boundary** (the M3 continuity requirement; DESIGN §5).

    Algorithm per block:
      1. level  = block RMS (floored by ``eps`` to avoid divide-by-zero)
      2. g_want = clip(target_rms / level, 0, max_gain)
      3. one-pole the gain toward g_want with an ASYMMETRIC time constant:
           * gain must DROP (signal got loud)  -> ``attack_s``  (fast: kills
             overshoot before it reaches the speaker)
           * gain must RISE (signal got quiet) -> ``release_s`` (slow: avoids
             "pumping" — gain chasing every dip)
         coefficient ``alpha = 1 - exp(-(n/fs)/tau)`` uses the ACTUAL block
         length ``n`` so the time constants stay correct under variable blocks.
      4. apply ``linspace(g_prev, g_new, n)`` to the block.

    Both *boost* and *attenuation* happen: a signal above the target is pulled
    DOWN (g_want < 1), a quiet one is pushed UP (g_want > 1, capped at
    ``max_gain``). During silence the gain climbs to ``max_gain`` and the noise
    floor rises with it — that is the documented physical limit (DESIGN §6 M3),
    NOT a bug; this stage does not gate noise.

    Parameters
    ----------
    fs : float
        Sample rate of the audio passing through (the audifier output, 48 kHz).
        Only used to turn attack/release seconds into per-block coefficients.
    target_rms : float
        Desired output RMS (audio is float in [-1, 1]; 0.2 ≈ comfortable).
    attack_s, release_s : float
        Time constants (seconds) for reducing / raising gain. attack ≪ release.
    max_gain : float
        Ceiling on the linear gain (default 12 ≈ +21.6 dB). Caps how far a quiet
        signal — and the noise floor under it — is lifted. Conservative by
        default: the old +40 dB ceiling lifted a quiet band / noise floor to a
        roar on real HW (post-M3 finding). Raise it (CLI --agc-max-gain) for
        more lift when you can tolerate more floor.
    eps : float
        Level floor; a pure divide-by-zero guard, NOT a noise gate.
    """

    def __init__(
        self,
        fs: float,
        *,
        target_rms: float = 0.2,
        attack_s: float = 0.010,
        release_s: float = 0.300,
        max_gain: float = 12.0,
        eps: float = 1e-6,
    ):
        if fs <= 0:
            raise ValueError(f"fs must be > 0, got {fs!r}")
        if target_rms <= 0:
            raise ValueError(f"target_rms must be > 0, got {target_rms!r}")
        if attack_s < 0 or release_s < 0:
            raise ValueError("attack_s / release_s must be >= 0")
        if max_gain <= 0:
            raise ValueError(f"max_gain must be > 0, got {max_gain!r}")
        self.fs = float(fs)
        self.target_rms = float(target_rms)
        self.attack_s = float(attack_s)
        self.release_s = float(release_s)
        self.max_gain = float(max_gain)
        self.eps = float(eps)
        # State carried across blocks. Start at unity so the very first samples
        # are not slammed; the AGC converges over the first attack/release.
        self._gain = 1.0

    @property
    def current_gain(self) -> float:
        """Current linear gain (for the status bar / tests)."""
        return self._gain

    def reset(self) -> None:
        """Drop the gain back to unity (e.g. after a band re-selection).

        NOT part of the frozen Protocol — an AGC-specific convenience. The worker
        does not require it (the AGC re-adapts on its own), but it lets a caller
        avoid carrying a loud band's gain into a newly selected quiet one."""
        self._gain = 1.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        n = x.size
        if n == 0:
            return x

        level = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        level = max(level, self.eps)
        g_want = self.target_rms / level
        if g_want > self.max_gain:
            g_want = self.max_gain

        # Asymmetric one-pole: fast when reducing gain, slow when raising it.
        tau = self.attack_s if g_want < self._gain else self.release_s
        if tau <= 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - np.exp(-(n / self.fs) / tau)
        g_new = self._gain + alpha * (g_want - self._gain)

        # Per-sample ramp -> continuous across the block boundary (no click).
        ramp = np.linspace(self._gain, g_new, n, dtype=np.float32)
        self._gain = float(g_new)
        return (x * ramp).astype(np.float32)
