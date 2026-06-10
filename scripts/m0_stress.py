"""M0 capture stress test — CLI (DESIGN §6 M0).

Examples (PowerShell — one command per line, no &&):
  python scripts\\m0_stress.py --source synthetic --tone-hz 45000 --duration 5
  python scripts\\m0_stress.py --source synthetic --duration 5 --load-ms 12   # force Xruns
  python scripts\\m0_stress.py --source wav --wav path\\to\\file.wav --duration 5
  python scripts\\m0_stress.py --source wasapi --duration 10 --blocksize 2048  # real HW (Kali)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly (python scripts\m0_stress.py) from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultrascan.capture.sources import make_source  # noqa: E402
from ultrascan.capture.stress import render_report, run_capture  # noqa: E402


def _force_utf8_stdout() -> None:
    # Console is cp932; force UTF-8 so the banner/report print cleanly (DESIGN §2).
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if callable(reconfig):
            reconfig(encoding="utf-8")


def list_devices() -> int:
    """Print input devices + host APIs so Kali can pick the UltraMic's WASAPI index."""
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    print("Host APIs:")
    for i, ha in enumerate(hostapis):
        print(f"  [{i}] {ha['name']}")
    print("\nInput devices (index | name | hostAPI | in-ch | default-rate):")
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] < 1:
            continue
        ha = hostapis[dev["hostapi"]]["name"]
        star = " <-- WASAPI" if "WASAPI" in ha.upper() else ""
        print(f"  {idx:>3} | {dev['name']} | {ha} | "
              f"{dev['max_input_channels']} | {dev['default_samplerate']:.0f}{star}")
    print("\nFor 250k exclusive capture, pass the WASAPI input index: "
          "--source wasapi --device <idx>")
    return 0


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
        return make_source("wav", path=args.wav, blocksize=args.blocksize)
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
    p = argparse.ArgumentParser(description="ultrascan M0 capture stress test")
    p.add_argument("--source", choices=["synthetic", "wav", "wasapi"], default="synthetic")
    p.add_argument("--duration", type=float, default=5.0, help="seconds to capture")
    p.add_argument("--samplerate", type=float, default=250_000.0)
    p.add_argument("--blocksize", type=int, default=2048)
    p.add_argument("--load-ms", type=float, default=0.0,
                   help="dummy per-callback sleep (overload boundary test)")
    p.add_argument("--tone-hz", type=float, default=45_000.0, help="synthetic tone freq")
    p.add_argument("--kind", choices=["tone", "chirp"], default="tone")
    p.add_argument("--wav", default=None, help="WAV path for --source wav")
    p.add_argument("--device", default=None, help="WASAPI device index or name substring")
    p.add_argument("--report", default="M0_capture_report.md", help="report output path")
    p.add_argument("--save-wav", default=None,
                   help="save the full capture to this WAV path (regression asset)")
    p.add_argument("--list-devices", action="store_true",
                   help="list audio input devices / host APIs and exit (helps pick WASAPI device)")
    args = p.parse_args(argv)

    if args.list_devices:
        return list_devices()

    source = build_source(args)
    if getattr(source, "is_synthetic", False):
        print("=== SYNTHETIC-ONLY: Sim source; real-HW judgment is Kali's (DESIGN §6) ===")
    print(f"[m0] source={source.name} rate={source.samplerate:.0f} "
          f"blocksize={source.blocksize} duration={args.duration}s load={args.load_ms}ms")

    result = run_capture(source, args.duration, load_ms=args.load_ms,
                         keep_samples=bool(args.save_wav))

    print(f"[m0] callbacks={result.n_callbacks} frames={result.total_frames} "
          f"xrun={result.overflow_count} "
          f"peak={result.peak_hz/1000:.2f}kHz "
          f"peak>24k={result.peak_above_guard_hz/1000:.2f}kHz "
          f">24kHz={result.energy_above_guard_frac*100:.1f}% "
          f"ultrasonic={'YES' if result.has_ultrasonic_energy else 'NO'}")

    if args.save_wav:
        if result.samples is not None and result.samples.size:
            import numpy as np
            from scipy.io import wavfile

            out = Path(args.save_wav)
            if out.parent and not out.parent.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(out), int(result.samplerate),
                          result.samples.astype(np.float32))
            print(f"[m0] wav written -> {out} "
                  f"({result.samples.size} samples @ {result.samplerate:.0f} Hz, float32)")
        else:
            print("[m0] --save-wav requested but no samples were captured (nothing written)")

    report = render_report(result)
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"[m0] report written -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
