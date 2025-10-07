#!/usr/bin/env python3
"""
run_smoke_test.py – helper to launch tier1_fixed.py in quick-smoke mode.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Tier-1 quick smoke test.")
    parser.add_argument(
        "--nproc",
        type=int,
        default=2,
        help="Number of GPUs/processes to use (passed to torchrun --nproc_per_node).",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29511,
        help="Port used by torchrun for rendezvous (maps to --master-port).",
    )
    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to tier1_fixed.py after --quick-smoke-test.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    script = repo_root / "tier1_fixed.py"
    if not script.is_file():
        print(f"Expected to find tier1_fixed.py next to this script (looked at {script})", file=sys.stderr)
        return 1

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={args.nproc}",
        f"--master-port={args.master_port}",
        str(script),
        "--quick-smoke-test",
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)

    print("Launching:", " ".join(cmd))
    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
