"""M4 GUI: event-detection overlay on the M2b band-select waterfall (DESIGN §6 M4).

Draws a box + measurement label per detected event on the scrolling waterfall.
GUI-thread only (DESIGN §2): the base ``_refresh`` QTimer also reads the detector
worker's thread-safe event snapshot and repositions a fixed pool of box/label
items. The detector runs on its own worker thread; this view only RENDERS, and
nothing here touches the audio path. Inherits the band drag-to-listen from
``BandSelectView`` so you can listen and watch detections at once.

Placement: each event's time is drawn RELATIVE to the detector's current time
(``t_now``) on the waterfall's [-span, 0] axis, so boxes scroll left with the
waterfall and drop off once older than the visible span. Frequency is the kHz
axis directly. Display-grade alignment (detector and display advance at the same
real-time rate); the numeric truth is the event log, not the pixel.
"""

from __future__ import annotations

import pyqtgraph as pg

from ultrascan.gui.band_view import BandSelectView


class EventOverlayView(BandSelectView):
    """BandSelectView + detected-event boxes/labels driven by a DetectorWorker."""

    def __init__(self, *args, detector_worker=None, max_boxes: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self._detector = detector_worker
        self._boxes = []
        self._labels = []
        box_pen = pg.mkPen((255, 80, 80), width=2)
        for _ in range(int(max_boxes)):
            box = pg.PlotCurveItem(pen=box_pen)
            box.setZValue(20)
            box.setVisible(False)
            self._wf_plot.addItem(box)
            label = pg.TextItem(color=(255, 225, 130), anchor=(0, 1))
            label.setZValue(21)
            label.setVisible(False)
            self._wf_plot.addItem(label)
            self._boxes.append(box)
            self._labels.append(label)

    # ── GUI-thread only ─────────────────────────────────────────────────────
    def _refresh(self) -> None:
        super()._refresh()
        if self._detector is None:
            return
        events, t_now = self._detector.snapshot()
        events = events[-len(self._boxes):]      # most recent fit the pool
        span = self._span_s
        for i, box in enumerate(self._boxes):
            label = self._labels[i]
            if i >= len(events):
                box.setVisible(False)
                label.setVisible(False)
                continue
            e = events[i]
            x0, x1 = e.t_start - t_now, e.t_end - t_now   # <= 0, scrolls left
            if x1 < -span:                                # fully scrolled off
                box.setVisible(False)
                label.setVisible(False)
                continue
            x0 = max(x0, -span)
            y0, y1 = e.f_min / 1e3, e.f_max / 1e3
            if y1 - y0 < 0.6:                             # keep a thin tone visible
                y0 -= 0.3
                y1 += 0.3
            box.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
            box.setVisible(True)
            label.setText(f"{e.f_peak / 1e3:.1f} kHz  {(e.t_end - e.t_start) * 1e3:.0f} ms")
            label.setPos(x0, y1)
            label.setVisible(True)
        self._status.showMessage(
            self._status.currentMessage() + f"   |   events {self._detector.n_events}"
        )
