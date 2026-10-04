"""Portable command-line entry point for the fourth-question calculations."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Q4 calculation; NPZ, CSV and JSON outputs")
    commands = parser.add_subparsers(dest="command", required=True)
    q42 = commands.add_parser("q4-2", help="Daily Q80 decisions with forecast prices")
    q42.add_argument("--smoke", action="store_true", help="31 warmup days plus two formal days")
    q43 = commands.add_parser("q4-3", help="Scheme-1 rolling two-stage decisions with forecast prices")
    q43.add_argument("--end-day", type=int, default=365,
                     help="Exclusive day index, 32..365; 32 computes February 1 only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.command == "q4-2":
        command = [sys.executable, "-B", "-u", str(root / "代码/run_q4_2.py")]
        if args.smoke:
            command.append("--smoke")
    else:
        if not 32 <= args.end_day <= 365:
            parser.error("--end-day must be 32..365")
        command = [sys.executable, "-B", "-u", str(root / "代码/run_q4_3.py"),
                   "--end-day", str(args.end_day)]
    return subprocess.call(command, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
