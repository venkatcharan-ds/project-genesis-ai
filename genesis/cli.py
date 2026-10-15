"""Command line entry point: `python -m genesis <command>`."""

from __future__ import annotations

import argparse
import sys

from genesis.config import load_config
from genesis.ledger import EvidenceLedger
from genesis.pipeline import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="genesis", description="GENESIS invention & engineering agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment from a YAML config")
    run.add_argument("--config", required=True)
    run.add_argument("--out", default="experiments/runs")
    run.add_argument("--ledger", default="experiments/ledger.jsonl")

    ver = sub.add_parser("verify-ledger", help="check the evidence ledger hash chain")
    ver.add_argument("--ledger", default="experiments/ledger.jsonl")

    args = parser.parse_args(argv)
    if args.command == "run":
        summary = run_experiment(load_config(args.config), args.out, args.ledger)
        print(f"run complete: {summary['run_dir']}")
        for name in summary["ranking_by_test_f1"]:
            ts = summary["results"][name]["test_tuned"]["timestep"]
            print(f"  {name:<24} P={ts['precision'] or 0:.3f} R={ts['recall'] or 0:.3f} "
                  f"FPR={ts['false_positive_rate'] or 0:.4f} F1={ts['f1']:.3f}")
        print("NOTE: synthetic data only — not physical-world validation.")
        return 0
    if args.command == "verify-ledger":
        ok, msg = EvidenceLedger(args.ledger).verify()
        print(msg)
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
