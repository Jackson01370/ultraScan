"""M4 live event detection + measurement (DESIGN §6 M4): adaptive-SNR detector
hung off the DISPLAY path, drawing event boxes on the waterfall while you listen.

The detector is an independent L1 consumer (its own ring cursor + STFT), so it
runs ALONGSIDE the M2b/M3 audio path without touching it (DESIGN §3 two-pass
separation). A steady tone (the room's ~25 kHz pest-repeller) is absorbed by the
adaptive background and NOT boxed; only newly risen sounds are.

Examples (PowerShell — one command per line, no &&):
  # Sim, headless: run the detector over a WAV and log measurements to CSV
  python scripts\\m4_detect.py --source wav --wav captures\\scene.wav --no-gui --no-audio --duration 8 --log-events captures\\events.csv
  # Sim synthetic 50 kHz tone, headless detection
  python scripts\\m4_detect.py --source synthetic --tone-hz 50000 --tone-amp 0.3 --no-gui --no-audio --duration 4
  # Real HW: live GUI, watch boxes appear on new sounds, drag the band to listen
  python scripts\\m4_detect.py --source wasapi --device 21 --blocksize 256 --log-events captures\\events.csv
  # Real HW headless soak with event log
  python scripts\\m4_detect.py --source wasapi --device 21 --no-gui --no-audio --duration 120 --log-events captures\\soak_events.csv

NB (DESIGN §4.4 / §11): this is a RULE/heuristic detector — its events are for
monitoring + measurement and must NOT be used as CNN training labels. The event
log is MEASUREMENTS (freq/duration/bandwidth/SNR), not audio (WAV recording is M5).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running directly (python scripts\m4_detect.py) from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultrascan.capture.ring_buffer import RingBuffer  # noqa: E402
from ultrascan.capture.sources import make_source  # noqa: E402

FS_OUT = 48_000.0


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            reconfig(encoding="utf-8")


def _ensure_qt_plugin_path() -> None:
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
            "synthetic", samplerate=args.samplerate, blocksize=args.blocksize,
            tone_hz=args.tone_hz, kind=args.kind, amplitude=args.tone_amp,
        )
    if args.source == "wav":
        if not args.wav:
            raise SystemExit("--source wav requires --wav PATH")
        return make_source("wav", path=args.wav, blocksize=args.blocksize, loop=args.loop)
    if args.source == "wasapi":
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        return make_source("wasapi", samplerate=args.samplerate,
                           blocksize=args.blocksize, device=device)
    raise SystemExit(f"unknown source: {args.source}")


def make_detector(stft, args):
    from ultrascan.detect.detector import AdaptiveSnrDetector

    return AdaptiveSnrDetector(
        stft.freqs_hz, stft.columns_per_second,
        snr_threshold_db=args.snr_db,
        bg_tau_s=args.bg_tau_ms / 1e3,
        bg_tau_active_s=args.bg_tau_active_s,
        min_run_bins=args.min_run_bins,
        hangover_s=args.hangover_ms / 1e3,
        min_event_s=args.min_event_ms / 1e3,
        warmup_s=args.warmup_ms / 1e3,
        f_min_hz=args.det_fmin_khz * 1e3,
    )


def fmt_event(e) -> str:
    ipi = "-" if e.ipi is None else f"{e.ipi * 1e3:.0f}ms"
    return (f"t=[{e.t_start:6.2f},{e.t_end:6.2f}]s dur={(e.t_end - e.t_start) * 1e3:5.0f}ms  "
            f"f_peak={e.f_peak / 1e3:6.2f}kHz f=[{e.f_min / 1e3:.1f},{e.f_max / 1e3:.1f}]kHz "
            f"bw={e.bandwidth / 1e3:5.2f}kHz slope={e.slope:+.3f}kHz/ms snr={e.snr_db:4.1f}dB "
            f"n={e.n_pulses} ipi={ipi}")


def main(argv=None) -> int:
    _force_utf8_stdout()
    p = argparse.ArgumentParser(description="ultrascan M4 live event detection + measurement")
    # capture / source (same knobs as m1/m2b)
    p.add_argument("--source", choices=["synthetic", "wav", "wasapi"], default="synthetic")
    p.add_argument("--samplerate", type=float, default=250_000.0)
    p.add_argument("--blocksize", type=int, default=256)
    p.add_argument("--tone-hz", type=float, default=50_000.0)
    p.add_argument("--tone-amp", type=float, default=0.3)
    p.add_argument("--kind", choices=["tone", "chirp"], default="tone")
    p.add_argument("--wav", default=None)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--device", default=None, help="WASAPI input device index or name substring")
    # display STFT (the detector uses the same nfft/hop so boxes line up)
    p.add_argument("--nfft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=None)
    p.add_argument("--history", type=int, default=512)
    p.add_argument("--levels", type=float, nargs=2, default=(-100.0, -20.0), metavar=("LO", "HI"))
    p.add_argument("--ring-seconds", type=float, default=4.0)
    # detector (DESIGN §4.4 method B — all tunable)
    p.add_argument("--snr-db", type=float, default=12.0, help="SNR over background to flag a bin")
    p.add_argument("--bg-tau-ms", type=float, default=500.0, help="background EMA tau, inactive bins")
    p.add_argument("--bg-tau-active-s", type=float, default=3.0, help="background tau, active bins (absorbs persistent tones)")
    p.add_argument("--min-run-bins", type=int, default=4, help="min contiguous active bins (rejects scattered noise)")
    p.add_argument("--hangover-ms", type=float, default=40.0, help="inactive gap bridged inside one event")
    p.add_argument("--min-event-ms", type=float, default=50.0, help="minimum event duration (rejects blips)")
    p.add_argument("--warmup-ms", type=float, default=300.0, help="suppress detection while background settles")
    p.add_argument("--det-fmin-khz", type=float, default=1.0, help="ignore bins below this (DC/low rumble)")
    p.add_argument("--log-events", default=None, help="write detected-event measurements to a CSV (OFF by default)")
    # audio (listen while detecting; reuses the M2b/M3 path). Off in headless by default.
    p.add_argument("--no-audio", action="store_true", help="do not start the audio path")
    p.add_argument("--f-lo", type=float, default=20_000.0, help="initial audio band lower edge [Hz]")
    p.add_argument("--bandwidth", type=float, default=10_000.0, help="initial audio bandwidth [Hz]")
    p.add_argument("--volume", type=float, default=1.0)
    p.add_argument("--agc", action="store_true", help="enable AGCGain on the audio path (M3)")
    p.add_argument("--agc-max-gain", type=float, default=12.0)
    p.add_argument("--sim-out", action="store_true", help="audio: speakerless wall-clock sim consumer")
    p.add_argument("--out-device", default=None)
    # run control
    p.add_argument("--no-gui", action="store_true", help="headless (Sim verification / HW soak)")
    p.add_argument("--duration", type=float, default=0.0, help="auto-quit after N seconds (0 = until close)")
    p.add_argument("--screenshot", default=None)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args(argv)

    if args.list_devices:
        from m0_stress import list_devices

        return list_devices()

    import numpy as np  # noqa: E402

    from ultrascan.detect.event_log import EventCsvLogger  # noqa: E402
    from ultrascan.detect.worker import DetectorWorker  # noqa: E402
    from ultrascan.dsp.stft import StftStream  # noqa: E402
    from ultrascan.gui.pipeline import RingWriter  # noqa: E402

    headless = args.no_gui
    want_audio = not args.no_audio and (not headless or args.sim_out)

    source = build_source(args)
    if getattr(source, "is_synthetic", False):
        print("=== SYNTHETIC-ONLY: Sim source; real-HW judgment is Kali's (DESIGN §6) ===")
    print("[m4] detector events are a RULE/heuristic output — NOT CNN training labels (DESIGN §11)")

    rate = source.samplerate
    ring = RingBuffer(int(rate * args.ring_seconds))
    writer = RingWriter(ring)

    # ── detection path: its OWN L1 cursor + STFT (decoupled from display/audio) ──
    det_stft = StftStream(rate, nfft=args.nfft, hop=args.hop)
    detector = make_detector(det_stft, args)

    collected = []
    logger = EventCsvLogger(args.log_events) if args.log_events else None

    def event_sink(ev):
        collected.append(ev)
        if logger is not None:
            logger(ev)

    det_worker = DetectorWorker(ring.reader(), det_stft, detector, event_sink=event_sink)

    print(f"[m4] source={source.name} rate={rate:.0f} nfft={det_stft.nfft} hop={det_stft.hop} "
          f"cols/s={det_stft.columns_per_second:.1f}  detector(snr={args.snr_db}dB "
          f"min_run={args.min_run_bins} min_event={args.min_event_ms:.0f}ms "
          f"bg_tau={args.bg_tau_ms:.0f}ms/{args.bg_tau_active_s:.1f}s fmin={args.det_fmin_khz}kHz)  "
          f"audio={'on' if want_audio else 'off'}  log={'on->' + args.log_events if logger else 'off'}")

    # ── optional audio path (listen while detecting; M2b/M3 modules) ────────────
    audio_worker = spsc = consumer = speaker = None
    if want_audio:
        from ultrascan.audio.output import SimPacedConsumer, SpeakerOutput  # noqa: E402
        from ultrascan.audio.spsc import SpscAudioRing  # noqa: E402
        from ultrascan.audio.worker import AudioWorker  # noqa: E402
        from ultrascan.dsp.audifier import HeterodyneAudifier  # noqa: E402

        gain = None
        if args.agc:
            from ultrascan.dsp.gain import AGCGain  # noqa: E402

            gain = AGCGain(FS_OUT, max_gain=args.agc_max_gain)
        spsc = SpscAudioRing(capacity=int(2.0 * FS_OUT), prebuffer=int(0.08 * FS_OUT))
        try:
            audio_worker = AudioWorker(
                ring.reader(), HeterodyneAudifier(), spsc, rate,
                f_lo_sel=args.f_lo, bandwidth=args.bandwidth, gain=gain, volume=args.volume,
            )
        except ValueError as exc:
            raise SystemExit(f"[m4] invalid initial band: {exc}")
        if args.sim_out:
            consumer = SimPacedConsumer(spsc, FS_OUT)
        else:
            out_device = args.out_device
            if out_device is not None and out_device.isdigit():
                out_device = int(out_device)
            speaker = SpeakerOutput(spsc, FS_OUT, device=out_device)

    t0 = time.perf_counter()

    def start_audio():
        if audio_worker is None:
            return
        audio_worker.start()
        (consumer or speaker).start()

    def stop_audio():
        if audio_worker is None:
            return
        audio_worker.stop()
        (consumer or speaker).stop()

    if headless:
        det_worker.start()
        start_audio()
        source.start(writer.on_block)
        try:
            end = time.perf_counter() + args.duration if args.duration > 0 else None
            while end is None or time.perf_counter() < end:
                time.sleep(0.5)
                if not source.is_running:
                    break
        except KeyboardInterrupt:
            print("[m4] interrupted")
        source.stop()
        stop_audio()
        det_worker.stop()
    else:
        _ensure_qt_plugin_path()
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

        from ultrascan.gui.event_view import EventOverlayView  # noqa: E402
        from ultrascan.gui.pipeline import DspWorker, make_display_queue  # noqa: E402

        disp_stft = StftStream(rate, nfft=args.nfft, hop=args.hop)
        queue = make_display_queue()
        dsp_worker = DspWorker(ring.reader(), disp_stft, queue)

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        view = EventOverlayView(
            disp_stft, queue, writer=writer, worker=dsp_worker,
            audio_worker=audio_worker, audio_ring=spsc, detector_worker=det_worker,
            band_khz=(args.f_lo / 1e3, (args.f_lo + args.bandwidth) / 1e3),
            history_cols=args.history, levels=tuple(args.levels),
            source_label=f"{source.name} @ {rate / 1e3:.0f} kHz + detect",
        )
        view.setWindowTitle(f"ultrascan M4 — detect (boxes = events) + drag to listen  [{source.name}]")
        view.show()

        dsp_worker.start()
        det_worker.start()
        start_audio()
        source.start(writer.on_block)

        shot = {"done": False}

        def take_shot():
            if not args.screenshot or shot["done"]:
                return
            out = Path(args.screenshot)
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
            view.screenshot(str(out))
            shot["done"] = True
            print(f"[m4] screenshot -> {out}")

        if args.duration > 0:
            QtCore.QTimer.singleShot(int(args.duration * 1000), lambda: (take_shot(), view.close()))
        try:
            app.exec_()
            take_shot()
        finally:
            source.stop()
            stop_audio()
            det_worker.stop()
            dsp_worker.stop()

    elapsed = time.perf_counter() - t0
    if logger is not None:
        logger.close()

    print(f"[m4] ---- run report ({elapsed:.1f} s) ----")
    print(f"[m4]   in_blocks={writer.n_blocks} in_xruns={writer.n_xruns} "
          f"det_dropped_samples={det_worker.n_dropped_samples} events={det_worker.n_events}")
    if want_audio and spsc is not None:
        print(f"[m4]   audio: q_underruns={spsc.n_underruns}")
    print(f"[m4]   detected {len(collected)} event(s):")
    for e in collected:
        print(f"[m4]     {fmt_event(e)}")
    if logger is not None:
        print(f"[m4]   event log -> {args.log_events} ({logger.n_written} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
