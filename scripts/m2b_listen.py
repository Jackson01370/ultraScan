"""M2b live audification — CLI (DESIGN §3: L0 -> L1 -> {L2 display, L3 audio} -> L0').

Examples (PowerShell — one command per line, no &&):
  python scripts\\m2b_listen.py --list-devices
  # Sim, no speakers, no GUI: numeric continuity/underrun verification
  python scripts\\m2b_listen.py --source synthetic --tone-hz 45000 --f-lo 40000 --no-gui --sim-out --duration 5 --save-out captures\\m2b_sim_tone.wav
  python scripts\\m2b_listen.py --source wav --wav captures\\m0_ultramic_keys_250k.wav --f-lo 20000 --no-gui --sim-out --duration 8 --save-out captures\\m2b_sim_wav.wav
  # Real HW + speakers + GUI band drag (device index moves: --list-devices first)
  python scripts\\m2b_listen.py --source wasapi --device 21 --blocksize 256 --f-lo 20000 --bandwidth 10000
  # Real HW headless underrun soak (minutes), output also saved for offline FFT
  python scripts\\m2b_listen.py --source wasapi --device 21 --blocksize 256 --no-gui --duration 180 --save-out captures\\m2b_realhw_out.wav

Click policy (a), decided for M2b: band re-selection resets DDC state -> an
audible click at the boundary is ACCEPTED; continuous sound has priority.
Crossfade/ramp smoothing is an M2c+ item, deliberately not implemented.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running the script directly (python scripts\m2b_listen.py) from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultrascan.capture.ring_buffer import RingBuffer  # noqa: E402
from ultrascan.capture.sources import make_source  # noqa: E402

FS_OUT = 48_000.0


def _force_utf8_stdout() -> None:
    # Console is cp932; force UTF-8 so device names / banners print cleanly (DESIGN §2).
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            reconfig(encoding="utf-8")


def _ensure_qt_plugin_path() -> None:
    # PyQt5 on this machine fails to self-locate its platform plugins (M1 finding).
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


def snapshot_stats(writer, worker, spsc, speaker=None, consumer=None) -> dict:
    """One coherent read of every health counter — taken BEFORE teardown, because
    stopping the source starves the chain by design and that is not the measurement."""
    s = {
        "in_blocks": writer.n_blocks,
        "in_xruns": writer.n_xruns,
        "audio_in_samples": worker.n_in_samples,
        "audio_out_samples": worker.n_out_samples,
        "audio_in_dropped": worker.n_dropped_in,
        "band_changes": worker.n_band_changes,
        "band": worker.band,
        "band_error": worker.last_band_error,
        "push_failed": worker.n_push_failed,
        "q_occupancy": spsc.occupancy,
        "q_underruns": spsc.n_underruns,
        "q_popped_real": spsc.n_popped_real,
        "q_popped_zero": spsc.n_popped_zero,
    }
    if speaker is not None:
        s["out_callbacks"] = speaker.n_callbacks
        s["out_pa_underflows"] = speaker.n_pa_underflows
    if consumer is not None:
        s["sim_emitted"] = consumer.n_emitted
    return s


def print_report(stats: dict, elapsed_s: float) -> None:
    print(f"[m2b] ---- run report ({elapsed_s:.1f} s) ----")
    for key, val in stats.items():
        print(f"[m2b]   {key} = {val}")
    verdict_q = "PASS" if stats["q_underruns"] == 0 else "FAIL"
    print(f"[m2b]   => queue underruns: {stats['q_underruns']}  [{verdict_q}]")
    if "out_pa_underflows" in stats:
        verdict_pa = "PASS" if stats["out_pa_underflows"] == 0 else "FAIL"
        print(f"[m2b]   => PortAudio output underflows: {stats['out_pa_underflows']}  [{verdict_pa}]")


def analyze_played(played, popped_zero: int, popped_real: int) -> None:
    """Offline numeric check of what was actually played (Sim or speaker capture)."""
    import numpy as np

    if played.size == 0:
        print("[m2b] analyze: nothing captured")
        return
    body = played[popped_zero:popped_zero + popped_real]
    if body.size < 1024:
        print(f"[m2b] analyze: body too short ({body.size} samples)")
        return
    spec = np.abs(np.fft.rfft(body * np.hanning(body.size)))
    peak_hz = float(np.argmax(spec)) * FS_OUT / body.size
    rms = float(np.sqrt(np.mean(body ** 2)))
    # Longest run of exact 0.0 inside the body — zero-fill gaps are exact zeros,
    # so this is a direct continuity probe (tone/repeller audio is never 0.0-flat).
    is_zero = (body == 0.0)
    if is_zero.any():
        edges = np.diff(is_zero.astype(np.int8))
        starts = np.flatnonzero(edges == 1) + 1
        ends = np.flatnonzero(edges == -1) + 1
        if is_zero[0]:
            starts = np.r_[0, starts]
        if is_zero[-1]:
            ends = np.r_[ends, is_zero.size]
        max_gap = int((ends - starts).max())
    else:
        max_gap = 0
    print(f"[m2b] analyze: played body {body.size} samples ({body.size / FS_OUT:.2f} s) "
          f"rms={rms:.4f} peak={peak_hz:.1f} Hz max_zero_run={max_gap}")


def save_wav(path: str, played) -> None:
    from scipy.io import wavfile

    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out), int(FS_OUT), played)
    print(f"[m2b] played audio written -> {out} ({played.size} samples @ 48 kHz)")


def main(argv=None) -> int:
    _force_utf8_stdout()
    p = argparse.ArgumentParser(description="ultrascan M2b live heterodyne audification")
    # capture side (same knobs as m1_view)
    p.add_argument("--source", choices=["synthetic", "wav", "wasapi"], default="synthetic")
    p.add_argument("--samplerate", type=float, default=250_000.0)
    p.add_argument("--blocksize", type=int, default=256,
                   help="capture blocksize (M0-soaked default 256; M2b re-measures under audio load)")
    p.add_argument("--tone-hz", type=float, default=45_000.0)
    p.add_argument("--kind", choices=["tone", "chirp"], default="tone")
    p.add_argument("--wav", default=None)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--device", default=None, help="WASAPI input device index or name substring")
    # audio path
    p.add_argument("--f-lo", type=float, default=20_000.0, help="initial band lower edge [Hz]")
    p.add_argument("--bandwidth", type=float, default=10_000.0, help="initial bandwidth [Hz]")
    p.add_argument("--volume", type=float, default=1.0,
                   help="fixed output attenuator (safety knob; NOT the M3 GainStage)")
    p.add_argument("--prebuffer-ms", type=float, default=80.0,
                   help="output read-ahead before sound starts (underrun headroom)")
    p.add_argument("--queue-s", type=float, default=2.0, help="SPSC queue capacity [s @ 48k]")
    p.add_argument("--out-device", default=None, help="output device index or name substring")
    p.add_argument("--out-blocksize", type=int, default=0, help="output blocksize (0 = PortAudio optimum)")
    p.add_argument("--sim-out", action="store_true",
                   help="no speakers: wall-clock-paced sim consumer (numeric verification)")
    p.add_argument("--save-out", default=None, help="write the played audio to a 48 kHz WAV")
    p.add_argument("--save-max-s", type=float, default=600.0,
                   help="cap for the speaker-path record buffer (preallocated)")
    # display side (GUI mode only; same knobs as m1_view)
    p.add_argument("--nfft", type=int, default=2048)
    p.add_argument("--hop", type=int, default=None)
    p.add_argument("--history", type=int, default=512)
    p.add_argument("--levels", type=float, nargs=2, default=(-100.0, -20.0), metavar=("LO", "HI"))
    p.add_argument("--ring-seconds", type=float, default=4.0)
    p.add_argument("--no-gui", action="store_true", help="headless (Sim verification / HW soak)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="auto-quit after N seconds (0 = run until window close / Ctrl+C)")
    p.add_argument("--screenshot", default=None)
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args(argv)

    if args.list_devices:
        from m0_stress import list_devices  # same dir; reuse the M0 lister

        return list_devices()

    import numpy as np  # noqa: E402  (after UTF-8 setup; needed below)

    from ultrascan.audio.output import SimPacedConsumer, SpeakerOutput  # noqa: E402
    from ultrascan.audio.spsc import SpscAudioRing  # noqa: E402
    from ultrascan.audio.worker import AudioWorker  # noqa: E402
    from ultrascan.dsp.audifier import HeterodyneAudifier  # noqa: E402
    from ultrascan.gui.pipeline import RingWriter  # noqa: E402

    source = build_source(args)
    if getattr(source, "is_synthetic", False):
        print("=== SYNTHETIC-ONLY: Sim source; real-HW judgment is Kali's (DESIGN §6) ===")
    print("[m2b] click policy (a): band re-selection clicks are accepted; "
          "crossfade/ramp smoothing deferred to M2c+ (recorded, not implemented)")

    rate = source.samplerate
    ring = RingBuffer(int(rate * args.ring_seconds))
    writer = RingWriter(ring)

    spsc = SpscAudioRing(
        capacity=int(args.queue_s * FS_OUT),
        prebuffer=int(args.prebuffer_ms / 1e3 * FS_OUT),
    )
    try:
        worker = AudioWorker(
            ring.reader(), HeterodyneAudifier(), spsc, rate,
            f_lo_sel=args.f_lo, bandwidth=args.bandwidth, volume=args.volume,
        )
    except ValueError as exc:
        raise SystemExit(f"[m2b] invalid initial band: {exc}")

    speaker = None
    consumer = None
    if args.sim_out:
        consumer = SimPacedConsumer(spsc, FS_OUT)
    else:
        out_device = args.out_device
        if out_device is not None and out_device.isdigit():
            out_device = int(out_device)
        record_s = args.save_max_s if (args.save_out or args.no_gui) else 0.0
        speaker = SpeakerOutput(
            spsc, FS_OUT, device=out_device, blocksize=args.out_blocksize,
            record_max_s=record_s,
        )

    print(f"[m2b] source={source.name} rate={rate:.0f} blocksize={source.blocksize}  "
          f"band={args.f_lo / 1e3:.1f}+{args.bandwidth / 1e3:.1f} kHz  out=48k "
          f"{'SIM' if args.sim_out else 'speaker'}  prebuffer={spsc.prebuffer} smp  "
          f"volume={args.volume}")

    t0 = time.perf_counter()
    stats = {}

    def _start_audio():
        worker.start()
        if consumer is not None:
            consumer.start()
        else:
            speaker.start()

    def _snapshot():
        nonlocal stats
        stats = snapshot_stats(writer, worker, spsc, speaker, consumer)

    def _teardown():
        source.stop()
        worker.stop()
        if consumer is not None:
            consumer.stop()
        else:
            speaker.stop()

    if args.no_gui:
        _start_audio()
        source.start(writer.on_block)
        try:
            end = time.perf_counter() + args.duration if args.duration > 0 else None
            while end is None or time.perf_counter() < end:
                time.sleep(0.5)
                if not source.is_running:  # e.g. WAV reached EOF without --loop
                    break
        except KeyboardInterrupt:
            print("[m2b] interrupted")
        _snapshot()
        _teardown()
    else:
        _ensure_qt_plugin_path()
        from pyqtgraph.Qt import QtCore, QtWidgets  # noqa: E402

        from ultrascan.dsp.stft import StftStream  # noqa: E402
        from ultrascan.gui.band_view import BandSelectView  # noqa: E402
        from ultrascan.gui.pipeline import DspWorker, make_display_queue  # noqa: E402

        stft = StftStream(rate, nfft=args.nfft, hop=args.hop)
        queue = make_display_queue()
        dsp_worker = DspWorker(ring.reader(), stft, queue)  # display's own L1 cursor

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        view = BandSelectView(
            stft, queue, writer=writer, worker=dsp_worker,
            audio_worker=worker, audio_ring=spsc,
            band_khz=(args.f_lo / 1e3, (args.f_lo + args.bandwidth) / 1e3),
            history_cols=args.history, levels=tuple(args.levels),
            source_label=f"{source.name} @ {rate / 1e3:.0f} kHz + audio",
        )
        view.setWindowTitle(f"ultrascan M2b — drag the band to listen  [{source.name}]")
        view.show()

        dsp_worker.start()
        _start_audio()
        source.start(writer.on_block)

        shot_done = {"written": False}

        def _take_screenshot():
            if not args.screenshot or shot_done["written"]:
                return
            out = Path(args.screenshot)
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
            view.screenshot(str(out))
            shot_done["written"] = True
            print(f"[m2b] screenshot written -> {out}")

        def _finish():
            _snapshot()
            _take_screenshot()
            view.close()

        if args.duration > 0:
            QtCore.QTimer.singleShot(int(args.duration * 1000), _finish)

        try:
            app.exec_()
            if not stats:
                _snapshot()  # manual close: snapshot before teardown
            _take_screenshot()
        finally:
            _teardown()
            dsp_worker.stop()

    elapsed = time.perf_counter() - t0
    print_report(stats, elapsed)

    played = consumer.output if consumer is not None else (
        speaker.recorded if speaker is not None else np.empty(0, dtype=np.float32)
    )
    if played.size:
        analyze_played(played, stats["q_popped_zero"], stats["q_popped_real"])
        if args.save_out:
            save_wav(args.save_out, played)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
