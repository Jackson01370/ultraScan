"""M1 live display — CLI (DESIGN §6 M1: L0 -> L1 -> L2 -> L4, no audio).

Examples (PowerShell — one command per line, no &&):
  python scripts\\m1_view.py --source synthetic --kind chirp
  python scripts\\m1_view.py --source wav --wav captures\\m0_ultramic_keys_250k.wav --loop
  python scripts\\m1_view.py --source wasapi --device 23 --blocksize 256
  python scripts\\m1_view.py --list-devices
  python scripts\\m1_view.py --source wav --wav captures\\m0_ultramic_keys_250k.wav --duration 6 --screenshot captures\\m1_wav.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly (python scripts\m1_view.py) from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultrascan.capture.ring_buffer import RingBuffer  # noqa: E402
from ultrascan.capture.sources import make_source  # noqa: E402


def _force_utf8_stdout() -> None:
    # Console is cp932; force UTF-8 so device names / banners print cleanly (DESIGN §2).
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            reconfig(encoding="utf-8")


def _ensure_qt_plugin_path() -> None:
    # On this machine PyQt5 fails to self-locate its platform plugins ("windows"
    # plugin not found, search path empty), so point Qt at the venv's copy.
    import os

    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return
    try:
        import PyQt5  # noqa: WPS433

        plugins = Path(PyQt5.__file__).parent / "Qt5" / "plugins" / "platforms"
        if plugins.is_dir():
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins)
    except ImportError:
        pass


def build_source(args):
    if args.source == "synthetic":
        return make_source(
            "synthetic",
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            tone_hz=args.tone_hz,
            kind=args.kind,
        )
    if args.source == "wav":
        if not args.wav:
            raise SystemExit("--source wav requires --wav PATH")
        return make_source("wav", path=args.wav, blocksize=args.blocksize, loop=args.loop)
    if args.source == "wasapi":
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        return make_source(
            "wasapi",
            samplerate=args.samplerate,
            blocksize=args.blocksize,
            device=device,
        )
    raise SystemExit(f"unknown source: {args.source}")


def main(argv=None) -> int:
    _force_utf8_stdout()
    p = argparse.ArgumentParser(description="ultrascan M1 live spectrum + waterfall")
    p.add_argument("--source", choices=["synthetic", "wav", "wasapi"], default="synthetic")
    p.add_argument("--samplerate", type=float, default=250_000.0)
    p.add_argument("--blocksize", type=int, default=256,
                   help="capture blocksize (M0-soaked default 256; provisional under load)")
    p.add_argument("--tone-hz", type=float, default=45_000.0, help="synthetic tone freq")
    p.add_argument("--kind", choices=["tone", "chirp"], default="tone")
    p.add_argument("--wav", default=None, help="WAV path for --source wav")
    p.add_argument("--loop", action="store_true", help="loop the WAV instead of stopping at EOF")
    p.add_argument("--device", default=None, help="WASAPI device index or name substring")
    p.add_argument("--nfft", type=int, default=2048, help="display STFT size (display-only knob)")
    p.add_argument("--hop", type=int, default=None, help="display STFT hop (default nfft//2)")
    p.add_argument("--history", type=int, default=512, help="waterfall width in columns")
    p.add_argument("--levels", type=float, nargs=2, default=(-100.0, -20.0),
                   metavar=("LO", "HI"), help="waterfall dBFS contrast levels (display gain)")
    p.add_argument("--ring-seconds", type=float, default=4.0, help="L1 ring capacity")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-quit after N seconds (0 = run until window closes)")
    p.add_argument("--screenshot", default=None,
                   help="save a PNG of the window before quitting (verification artifact)")
    p.add_argument("--list-devices", action="store_true",
                   help="list audio input devices / host APIs and exit")
    args = p.parse_args(argv)

    if args.list_devices:
        from m0_stress import list_devices  # same dir; reuse the M0 lister

        return list_devices()

    source = build_source(args)
    if getattr(source, "is_synthetic", False):
        print("=== SYNTHETIC-ONLY: Sim source; real-HW judgment is Kali's (DESIGN §6) ===")

    # Import Qt only past arg parsing so --list-devices stays GUI-free.
    _ensure_qt_plugin_path()
    from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

    from ultrascan.dsp.stft import StftStream  # noqa: E402
    from ultrascan.gui.app import LiveView  # noqa: E402
    from ultrascan.gui.pipeline import DspWorker, RingWriter, make_display_queue  # noqa: E402

    rate = source.samplerate
    ring = RingBuffer(int(rate * args.ring_seconds))
    writer = RingWriter(ring)
    stft = StftStream(rate, nfft=args.nfft, hop=args.hop)
    queue = make_display_queue()
    worker = DspWorker(ring.reader(), stft, queue)

    print(f"[m1] source={source.name} rate={rate:.0f} blocksize={source.blocksize} "
          f"nfft={stft.nfft} hop={stft.hop} cols/s={stft.columns_per_second:.1f} "
          f"history={args.history} levels={tuple(args.levels)}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    view = LiveView(
        stft, queue, writer=writer, worker=worker,
        history_cols=args.history, levels=tuple(args.levels),
        source_label=f"{source.name} @ {rate / 1e3:.0f} kHz",
    )
    view.show()

    worker.start()
    source.start(writer.on_block)

    shot_done = {"written": False}

    def _take_screenshot():
        # Idempotent: runs from the timed path AND after a manual window close,
        # so --screenshot is honored either way (review finding: it used to be
        # silently ignored without --duration).
        if not args.screenshot or shot_done["written"]:
            return
        out = Path(args.screenshot)
        if out.parent and not out.parent.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
        view.screenshot(str(out))
        shot_done["written"] = True
        print(f"[m1] screenshot written -> {out}")

    def _finish():
        _take_screenshot()
        view.close()

    if args.duration > 0:
        QtCore.QTimer.singleShot(int(args.duration * 1000), _finish)

    try:
        app.exec_()
        _take_screenshot()  # manual-close path: window hidden but still renderable
    finally:
        source.stop()
        worker.stop()

    print(f"[m1] done: blocks={writer.n_blocks} xrun={writer.n_xruns} "
          f"cols={worker.n_columns} dropped_samples={worker.n_dropped_samples}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
