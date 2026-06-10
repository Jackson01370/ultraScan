"""L4 GUI: live spectrum + horizontally scrolling waterfall (DESIGN §3, M1).

Qt widgets are touched ONLY from the GUI thread (DESIGN §2): a QTimer running
on the GUI thread drains the display deque that ``DspWorker`` fills. Display is
latest-priority and may drop frames; nothing here feeds the audio path.

Display gain (waterfall contrast / dB levels) is a *display* knob — kept
separate from the future audio gain (DESIGN §4.3). M1 exposes it as fixed CLI
defaults only.
"""

from __future__ import annotations

from typing import Deque, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ultrascan.dsp.stft import StftStream
from ultrascan.gui.pipeline import DspWorker, RingWriter

pg.setConfigOptions(imageAxisOrder="col-major")  # image[axis0=x(time), axis1=y(freq)]


def _lookup_table():
    for name in ("inferno", "viridis", "CET-L9"):
        try:
            return pg.colormap.get(name).getLookupTable(nPts=256)
        except Exception:  # noqa: BLE001 - colormap availability varies per install
            continue
    ramp = np.linspace(0, 255, 256, dtype=np.uint8)
    return np.stack([ramp, ramp, ramp], axis=1)  # grayscale fallback


class LiveView(QtWidgets.QMainWindow):
    """Live one-sided spectrum (top) + scrolling waterfall (bottom)."""

    def __init__(
        self,
        stft: StftStream,
        queue: "Deque[np.ndarray]",
        writer: Optional[RingWriter] = None,
        worker: Optional[DspWorker] = None,
        history_cols: int = 512,
        levels: Tuple[float, float] = (-100.0, -20.0),
        refresh_ms: int = 33,
        source_label: str = "",
    ):
        super().__init__()
        self._stft = stft
        self._queue = queue
        self._writer = writer
        self._worker = worker
        self._levels = (float(levels[0]), float(levels[1]))
        self._history = int(history_cols)
        self._n_bins = stft.n_bins
        self._freqs_khz = stft.freqs_hz / 1e3
        self._img = np.full((self._history, self._n_bins), self._levels[0], dtype=np.float32)
        self._latest_col: Optional[np.ndarray] = None
        self._span_s = self._history / stft.columns_per_second

        self.setWindowTitle(f"ultrascan M1 — live spectrum + waterfall  [{source_label}]")
        self.resize(1100, 750)

        central = pg.GraphicsLayoutWidget()
        self.setCentralWidget(central)

        nyq_khz = float(self._freqs_khz[-1])
        self._spec_plot = central.addPlot(row=0, col=0)
        self._spec_plot.setLabel("bottom", "frequency", units="kHz")
        self._spec_plot.setLabel("left", "magnitude", units="dBFS")
        self._spec_plot.setXRange(0.0, nyq_khz, padding=0)
        self._spec_plot.setYRange(self._levels[0] - 10.0, 0.0, padding=0)
        self._spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self._spec_curve = self._spec_plot.plot(pen=pg.mkPen("#40c4ff", width=1))

        self._wf_plot = central.addPlot(row=1, col=0)
        self._wf_plot.setLabel("bottom", "time", units="s")
        self._wf_plot.setLabel("left", "frequency", units="kHz")
        self._wf_item = pg.ImageItem()
        self._wf_item.setLookupTable(_lookup_table())
        self._wf_item.setLevels(self._levels)
        self._wf_plot.addItem(self._wf_item)
        self._wf_plot.setXRange(-self._span_s, 0.0, padding=0)
        self._wf_plot.setYRange(0.0, nyq_khz, padding=0)
        central.ci.layout.setRowStretchFactor(0, 1)
        central.ci.layout.setRowStretchFactor(1, 2)

        self._status = self.statusBar()
        self._wf_item.setImage(self._img, autoLevels=False)
        self._wf_item.setRect(QtCore.QRectF(-self._span_s, 0.0, self._span_s, nyq_khz))

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(int(refresh_ms))

    # ── GUI-thread only ─────────────────────────────────────────────────────
    def _drain(self) -> int:
        """Move all pending column batches from the worker deque into the image."""
        batches = []
        while True:
            try:
                batches.append(self._queue.popleft())
            except IndexError:
                break
        if not batches:
            return 0
        cols = np.concatenate(batches, axis=0) if len(batches) > 1 else batches[0]
        k = cols.shape[0]
        if k >= self._history:
            self._img[:] = cols[-self._history:]
        else:
            self._img = np.roll(self._img, -k, axis=0)
            self._img[-k:] = cols
        self._latest_col = cols[-1]
        return k

    def _refresh(self) -> None:
        if self._drain():
            self._wf_item.setImage(self._img, autoLevels=False, levels=self._levels)
            if self._latest_col is not None:
                self._spec_curve.setData(self._freqs_khz, self._latest_col)
        parts = []
        if self._writer is not None:
            parts.append(f"blocks={self._writer.n_blocks}  xrun={self._writer.n_xruns}")
        if self._worker is not None:
            parts.append(
                f"cols={self._worker.n_columns}  dropped={self._worker.n_dropped_samples}"
            )
        parts.append(f"span={self._span_s:.1f}s  levels={self._levels} dBFS")
        self._status.showMessage("   ".join(parts))

    def screenshot(self, path: str) -> None:
        """Save the current window contents as a PNG (verification artifact)."""
        self.grab().save(path)
