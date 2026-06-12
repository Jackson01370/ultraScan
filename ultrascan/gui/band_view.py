"""M2b GUI: M1 LiveView + draggable band selection on the waterfall (DESIGN §3 L4).

A horizontal ``LinearRegionItem`` on the waterfall selects [f_lo, f_hi] (kHz on
the frequency axis). Dragging it (or its edges) maps lower edge -> ``f_lo_sel``
and height -> ``bandwidth``, forwarded to the audio worker via
``request_band()`` — the worker applies ``configure()`` on its own thread, so
Qt widgets stay GUI-thread-only and the frozen audifier stays single-threaded
(DESIGN §2). A selection the audifier refuses (atomic configure) keeps the old
band playing; the refusal shows up in the status bar instead of crashing.
"""

from __future__ import annotations

import pyqtgraph as pg

from ultrascan.gui.app import LiveView


class BandSelectView(LiveView):
    """LiveView plus the audio band selector + audio-path status readout."""

    def __init__(
        self,
        *args,
        audio_worker=None,
        audio_ring=None,
        band_khz=(20.0, 30.0),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._audio_worker = audio_worker
        self._audio_ring = audio_ring

        nyq_khz = float(self._freqs_khz[-1])
        self._region = pg.LinearRegionItem(
            values=(float(band_khz[0]), float(band_khz[1])),
            orientation="horizontal",  # two horizontal lines bounding a freq band
            brush=(64, 196, 255, 40),
            movable=True,
        )
        self._region.setBounds((0.0, nyq_khz))
        self._region.setZValue(10)
        self._wf_plot.addItem(self._region)
        self._region.sigRegionChangeFinished.connect(self._on_band_changed)

    # ── GUI-thread only ─────────────────────────────────────────────────────
    def _on_band_changed(self) -> None:
        if self._audio_worker is None:
            return
        lo_khz, hi_khz = self._region.getRegion()
        self._audio_worker.request_band(lo_khz * 1e3, (hi_khz - lo_khz) * 1e3)

    def _refresh(self) -> None:
        super()._refresh()
        w = self._audio_worker
        if w is None:
            return
        lo, bw = w.band
        parts = [f"audio: {lo / 1e3:.1f}–{(lo + bw) / 1e3:.1f} kHz"]
        ring = self._audio_ring
        if ring is not None:
            parts.append(f"buf {ring.occupancy}/{ring.capacity}")
            parts.append(f"underruns {ring.n_underruns}")
        if w.n_dropped_in:
            parts.append(f"in-drop {w.n_dropped_in}")
        if w.last_band_error:
            parts.append(f"REFUSED: {w.last_band_error[:70]}")
        self._status.showMessage(
            self._status.currentMessage() + "   |   " + "  ".join(parts)
        )
