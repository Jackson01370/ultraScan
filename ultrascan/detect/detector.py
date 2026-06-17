"""FROZEN CONTRACT §4.4 — Detector Protocol + adaptive-SNR detector (M4).

╔══════════════════════════════════════════════════════════════════════════╗
║  FROZEN at M4.  The FROZEN thing is the SIGNATURE: the ``Detector``        ║
║  Protocol (``detect(frame_or_buffer) -> List[Event]``) and the ``Event``   ║
║  field set (events.py).  Signature changes need human approval (§4, §11).  ║
║  Tuning ``AdaptiveSnrDetector``'s default params is NOT a signature change ║
║  (allowed with approval — same scope rule as gain.py).                     ║
╚══════════════════════════════════════════════════════════════════════════╝

Later detectors (e.g. a CNN classifier) are NEW classes / modules and do NOT
re-edit this frozen signature (same discipline as ``AGCGain`` @ M3,
``HeterodyneAudifier`` @ M2a — the single M4 implementation lands here, then the
file freezes).

Rule (§4.4, §11): a rule/heuristic detector's output must NOT be used as CNN
training labels (label-noise prevention; sigscan discipline). ``AdaptiveSnrDetector``
is exactly such a heuristic detector — its events are for monitoring/measurement.

Method B — adaptive noise-floor SNR (per the M4 work order):
  * Per frequency bin, keep a running background estimate (an EMA of the dB
    spectrum). A bin whose current level exceeds its background by more than
    ``snr_threshold_db`` is "active".
  * The background ADAPTS, so a steady tone (e.g. the room's ~25 kHz ultrasonic
    pest-repeller) is absorbed into the background and is NOT detected — only a
    NEWLY risen sound stands out. Adaptive (not fixed) threshold is the whole
    point: it works whether the repeller is on or off.
  * Background-contamination guard: while a bin is active its background adapts
    much MORE SLOWLY (``bg_tau_active_s`` ≫ ``bg_tau_s``), so a transient event
    does not pull its own background up and self-cancel — yet a *persistent* new
    tone is still absorbed after a few seconds instead of being flagged forever.

Noise rejection (why scattered noise does NOT fire):
  A genuine source is SPECTRALLY CONCENTRATED (energy in a contiguous run of
  bins) and PERSISTENT in time. Random noise momentarily pokes single, scattered
  bins above threshold — never a contiguous run that holds still across frames.
  So a frame's activity is taken as contiguous active RUNS (≥ ``min_run_bins``),
  and runs are tracked across frames by frequency OVERLAP into events; a track
  shorter than ``min_event_s`` is discarded (a noise blip), a lasting one is
  emitted and measured (peak / min / max freq, bandwidth, duration, slope, SNR).
  Distinct simultaneous sources at different frequencies form distinct tracks.

Granularity: detection runs on the display-side STFT frames (hop-spaced), so the
time resolution is one STFT hop. Full capture of very short pulses (bat clicks,
a few ms) is explicitly deferred to future tuning (M4 work order).
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .events import Event

# A per-frame detection:
#   (lo_bin, hi_bin, peak_bin, peak_db, max_snr_db, meas_lo_bin, meas_hi_bin)
# lo..hi is the SNR-threshold extent (used for tracking/overlap); meas_lo..meas_hi
# is the tighter -``meas_db_below_peak`` extent used for the frequency MEASUREMENT
# (so a loud tone's wide sidelobe skirt does not inflate its reported bandwidth).
_Run = Tuple[int, int, int, float, float, int, int]


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame_or_buffer) -> List[Event]:
        """SNR / energy based event detection + segmentation."""
        ...


class _Track:
    """O(1)-memory accumulator for an in-progress event (one frequency cluster).

    Frequency extent / peak are running scalars; the peak-frequency-vs-time
    linear fit (slope) is kept as running sums, so an event of any duration costs
    constant memory. ``cur_lo/cur_hi`` is the most recent frame's bin span, used
    to match the next frame's runs by frequency overlap (lets a chirp drift).
    """

    __slots__ = (
        "start_col", "last_active_col", "cur_lo", "cur_hi",
        "peak_db", "peak_bin", "pkbin_lo", "pkbin_hi", "best_lo", "best_hi",
        "max_snr", "n", "st", "sf", "stt", "stf",
    )

    def __init__(self, col: int):
        self.start_col = col
        self.last_active_col = col
        self.cur_lo = 0
        self.cur_hi = 0
        self.peak_db = -np.inf
        self.peak_bin = 0
        self.pkbin_lo = 1 << 30     # peak-frequency excursion (robust to transients)
        self.pkbin_hi = -1
        self.best_lo = 1 << 30      # -meas_db extent at the single strongest frame
        self.best_hi = -1
        self.max_snr = -np.inf
        self.n = 0                  # running sums for the slope least-squares fit
        self.st = 0.0
        self.sf = 0.0
        self.stt = 0.0
        self.stf = 0.0


class AdaptiveSnrDetector:
    """Adaptive noise-floor SNR detector (DESIGN §4.4, method B).

    Construct with the STFT axes it will consume:
      * ``freqs_hz``           — bin centre frequencies (``StftStream.freqs_hz``)
      * ``columns_per_second`` — STFT column rate (``StftStream.columns_per_second``)

    Then feed dBFS columns to :meth:`detect` (the same array ``StftStream.push``
    returns: shape ``(n_cols, n_bins)``). It is stateful and streaming — the
    background and any in-progress tracks persist across calls; :meth:`detect`
    returns the events that COMPLETED during the call, and :meth:`finalize`
    closes tracks still open at end-of-stream.

    Parameters (all tunable; defaults are a monitoring-grade starting point)
    ----------
    snr_threshold_db : float
        How far (dB) a bin must exceed its background to count as active.
    bg_tau_s : float
        Background EMA time constant for INACTIVE bins (fast: tracks the floor).
    bg_tau_active_s : float
        Background EMA time constant for ACTIVE bins (slow: avoids self-cancel;
        absorbs a persistent new tone after a few seconds). Should be ≫ bg_tau_s.
    min_run_bins : int
        Minimum CONTIGUOUS active bins for a frame detection (spectral
        concentration; rejects scattered single-bin noise).
    run_gap_bins : int
        Inactive bins bridged when forming a run, so a strong tone's mainlobe and
        its Hann sidelobes (split by spectral nulls) count as ONE run/peak.
    meas_db_below_peak : float
        Frequency extent (f_min/f_max/bandwidth) is measured over bins within
        this many dB of the event peak — NOT the full SNR-threshold extent — so a
        very loud tone's window sidelobe skirt does not inflate its bandwidth.
    track_tol_bins : int
        Frequency drift (bins) tolerated when matching a run to an open track
        across frames — lets a chirp's peak move and still continue one event.
    hangover_s : float
        Inactive gap tolerated inside one event (bridges brief dropouts).
    min_event_s : float
        Minimum event duration; shorter tracks are discarded (noise blips).
    warmup_s : float
        After (re)start, suppress event detection this long while the background
        settles. The background still adapts during warmup.
    f_min_hz : float
        Ignore bins below this frequency (e.g. DC / very-low rumble). 0 = all.
    """

    def __init__(
        self,
        freqs_hz: np.ndarray,
        columns_per_second: float,
        *,
        snr_threshold_db: float = 12.0,
        bg_tau_s: float = 0.5,
        bg_tau_active_s: float = 3.0,
        min_run_bins: int = 4,
        run_gap_bins: int = 2,
        meas_db_below_peak: float = 20.0,
        track_tol_bins: int = 4,
        hangover_s: float = 0.04,
        min_event_s: float = 0.05,
        warmup_s: float = 0.3,
        f_min_hz: float = 0.0,
    ):
        freqs = np.asarray(freqs_hz, dtype=np.float64).reshape(-1)
        if freqs.size < 2:
            raise ValueError("freqs_hz must have >= 2 bins")
        if columns_per_second <= 0:
            raise ValueError("columns_per_second must be > 0")
        if snr_threshold_db <= 0:
            raise ValueError("snr_threshold_db must be > 0")
        if min_run_bins < 1:
            raise ValueError("min_run_bins must be >= 1")
        if meas_db_below_peak <= 0:
            raise ValueError("meas_db_below_peak must be > 0")
        self.freqs_hz = freqs
        self.n_bins = int(freqs.size)
        self.cps = float(columns_per_second)
        self.hop_s = 1.0 / self.cps
        self.snr_threshold_db = float(snr_threshold_db)
        self.min_run_bins = int(min_run_bins)
        self.run_gap_bins = max(0, int(run_gap_bins))
        self.meas_db = float(meas_db_below_peak)
        self.track_tol_bins = int(track_tol_bins)
        self._a_inactive = self._alpha(bg_tau_s)
        self._a_active = self._alpha(bg_tau_active_s)
        self.hangover_cols = max(0, int(round(hangover_s * self.cps)))
        self.min_event_cols = max(1, int(round(min_event_s * self.cps)))
        self.warmup_cols = max(0, int(round(warmup_s * self.cps)))
        self._analysis_bins = freqs >= float(f_min_hz)
        self._col = 0
        self.reset()

    def _alpha(self, tau_s: float) -> float:
        tau = float(tau_s)
        if tau < 0:
            raise ValueError("time constant must be >= 0")
        if tau == 0.0:
            return 1.0
        return float(1.0 - np.exp(-self.hop_s / tau))

    @property
    def t_now(self) -> float:
        """Absolute time (s) of the most recent column processed."""
        return self._col * self.hop_s

    @property
    def background_db(self) -> Optional[np.ndarray]:
        """Current per-bin background estimate (None before the first column)."""
        return None if self._bg is None else self._bg.copy()

    def reset(self) -> None:
        """Clear detection state (background + open tracks + warmup countdown).

        The absolute time cursor (``t_now`` / event times) is preserved, so an
        L1 reader gap that forces a reset does not desync overlay placement
        (event time and ``t_now`` share the cursor — the relative offset the GUI
        uses is unaffected). NOT part of the frozen Protocol."""
        self._bg: Optional[np.ndarray] = None
        self._since_reset = 0
        self._open: List[_Track] = []

    def detect(self, frame_or_buffer) -> List[Event]:
        """Consume dBFS column(s); return events COMPLETED during this call."""
        cols = np.asarray(frame_or_buffer, dtype=np.float32)
        if cols.ndim == 1:
            cols = cols[None, :]
        if cols.ndim != 2 or cols.shape[1] != self.n_bins:
            raise ValueError(
                f"expected (n_cols, {self.n_bins}) dB columns, got {cols.shape}"
            )
        out: List[Event] = []
        for row in cols:
            out.extend(self._step(row.astype(np.float64)))
        return out

    def finalize(self) -> List[Event]:
        """Close tracks still open at end-of-stream (worker stop / test end)."""
        out = [ev for ev in (self._close(tr) for tr in self._open) if ev is not None]
        self._open = []
        return out

    # ── internals ────────────────────────────────────────────────────────────
    def _step(self, x: np.ndarray) -> List[Event]:
        self._col += 1
        self._since_reset += 1
        if self._bg is None:               # lazy init: first frame defines the floor
            self._bg = x.copy()
            return []

        snr = x - self._bg                  # judged against the PAST background
        active = (snr > self.snr_threshold_db) & self._analysis_bins
        # Update background AFTER the SNR decision: inactive bins track fast,
        # active bins creep slowly (anti-self-cancel; absorbs persistent tones).
        alpha = np.where(active, self._a_active, self._a_inactive)
        self._bg += alpha * (x - self._bg)

        if self._since_reset > self.warmup_cols:
            runs = self._find_runs(active, x, snr)
            if runs:
                # Only the strongest run (the dominant source's mainlobe) is
                # tracked per frame: this ignores a loud tone's weaker sidelobe
                # runs (so they neither spawn events nor inflate bandwidth) and
                # keeps the measurement tied to the true peak. Trade-off: at any
                # instant the dominant source wins; a strictly simultaneous
                # weaker source is not separately tracked (M4 accepted limit —
                # sequential events at different frequencies ARE separated).
                self._assign(max(runs, key=lambda r: r[3]))

        # Close tracks gone inactive longer than the hangover bridge.
        closed: List[Event] = []
        survivors: List[_Track] = []
        for tr in self._open:
            if self._col - tr.last_active_col > self.hangover_cols:
                ev = self._close(tr)
                if ev is not None:
                    closed.append(ev)
            else:
                survivors.append(tr)
        self._open = survivors
        return closed

    def _find_runs(self, active, x, snr) -> List[_Run]:
        """Contiguous active runs of length >= min_run_bins.

        Each run carries both its full SNR-threshold extent (lo..hi, for
        tracking) and the tighter -meas_db extent around the peak (for the
        bandwidth measurement)."""
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return []
        # Bridge small spectral nulls: a strong tone's mainlobe + Hann sidelobes
        # are split by 1-2 inactive bins; merging them gives ONE run with ONE
        # (mainlobe) peak, so the -meas_db measurement excludes the sidelobes.
        splits = np.flatnonzero(np.diff(idx) > self.run_gap_bins + 1) + 1
        runs: List[_Run] = []
        for r in np.split(idx, splits):
            if r.size < self.min_run_bins:
                continue
            xr = x[r]
            pk = int(r[int(np.argmax(xr))])
            pkdb = float(x[pk])
            keep = r[xr >= pkdb - self.meas_db]   # measurement extent (>= peak - meas_db)
            runs.append((int(r[0]), int(r[-1]), pk, pkdb, float(snr[r].max()),
                         int(keep[0]), int(keep[-1])))
        return runs

    def _assign(self, run: _Run) -> None:
        """Extend the best frequency-overlapping open track, or start a new one."""
        lo, hi, pk, _pkdb, _snr, _lo_m, _hi_m = run
        tol = self.track_tol_bins
        best: Optional[_Track] = None
        best_dist = None
        for tr in self._open:
            if hi >= tr.cur_lo - tol and lo <= tr.cur_hi + tol:
                dist = abs(pk - 0.5 * (tr.cur_lo + tr.cur_hi))
                if best is None or dist < best_dist:
                    best, best_dist = tr, dist
        if best is None:
            best = _Track(self._col)
            self._open.append(best)
        self._extend(best, run)

    def _extend(self, tr: _Track, run: _Run) -> None:
        lo, hi, pk, pkdb, msnr, lo_m, hi_m = run
        tr.last_active_col = self._col
        tr.cur_lo, tr.cur_hi = lo, hi          # full extent drives cross-frame tracking
        # Peak-frequency excursion (robust to onset/offset transient smear: the
        # peak bin stays on the source even when a transient frame is broadband).
        if pk < tr.pkbin_lo:
            tr.pkbin_lo = pk
        if pk > tr.pkbin_hi:
            tr.pkbin_hi = pk
        # Instantaneous -meas_db extent, captured at the single STRONGEST frame —
        # a steady frame, never a transient, so a tone stays narrow.
        if pkdb > tr.peak_db:
            tr.peak_db, tr.peak_bin = pkdb, pk
            tr.best_lo, tr.best_hi = lo_m, hi_m
        if msnr > tr.max_snr:
            tr.max_snr = msnr
        t = self._col * self.hop_s
        f = float(self.freqs_hz[pk])
        tr.n += 1
        tr.st += t
        tr.sf += f
        tr.stt += t * t
        tr.stf += t * f

    def _close(self, tr: _Track) -> Optional[Event]:
        if tr.last_active_col - tr.start_col + 1 < self.min_event_cols:
            return None
        # Frequency extent = peak-frequency excursion (sweep) UNION the strongest
        # frame's spectral width (instantaneous bandwidth) — robust to transients.
        fmin_bin = min(tr.pkbin_lo, tr.best_lo)
        fmax_bin = max(tr.pkbin_hi, tr.best_hi)
        return Event(
            t_start=tr.start_col * self.hop_s,
            t_end=(tr.last_active_col + 1) * self.hop_s,
            f_peak=float(self.freqs_hz[tr.peak_bin]),
            f_min=float(self.freqs_hz[fmin_bin]),
            f_max=float(self.freqs_hz[fmax_bin]),
            bandwidth=float(self.freqs_hz[fmax_bin] - self.freqs_hz[fmin_bin]),
            slope=self._slope_khz_per_ms(tr),
            n_pulses=1,
            ipi=None,
            snr_db=float(tr.max_snr),
        )

    def _slope_khz_per_ms(self, tr: _Track) -> float:
        if tr.n < 2:
            return 0.0
        denom = tr.n * tr.stt - tr.st * tr.st
        if denom <= 0.0:
            return 0.0
        slope_hz_per_s = (tr.n * tr.stf - tr.st * tr.sf) / denom
        return float(slope_hz_per_s / 1e6)   # Hz/s -> kHz/ms
