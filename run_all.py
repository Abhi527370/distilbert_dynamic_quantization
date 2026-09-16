"""
run_all.py  -  ONE command that reproduces every experiment.

    python run_all.py                  # full run on the 872 dev sentences
    python run_all.py --smoke          # 64 sentences, only to check that everything works
    python run_all.py --threads 2 --results_dir results_2threads   # optional extra run

Steps:
  0) preflight: check this machine supports the requested methods
     save library versions + hardware          -> results/environment.json
  1) count MACs once on the FP32 model          -> results/macs.json
  2) benchmark every method in a FRESH process  -> results/<method>.json
  3) build tables and figures for the report    -> results/summary.md, *.png
"""
import argparse
import os
import subprocess
import sys
import time

from src import config
from src.quantization import probe_method, select_quantized_engine

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def preflight(methods):
    """Check that this machine supports every requested method before measuring."""
    engine = select_quantized_engine()
    print(f"Quantized engine: {engine}")
    failed = []
    for method in methods:
        ok, reason = probe_method(method)
        if not ok:
            failed.append((method, reason))
    if failed:
        print("\nThis machine cannot run some of the requested methods:\n")
        for method, reason in failed:
            print(f"  {method}\n    {reason}\n")
        print("Run 'python check_setup.py' - it tests every method and prints a")
        print("working command for this machine.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run all DistilBERT quantization experiments")
    parser.add_argument("--threads", type=int, default=config.NUM_THREADS)
    parser.add_argument("--methods", nargs="+", default=config.METHODS,
                        choices=config.ALL_METHODS,
                        help="default: FP32 + the 2 quantized variants required by the task")
    parser.add_argument("--smoke", action="store_true", help="quick check on 64 sentences")
    parser.add_argument("--results_dir", default=None)
    args = parser.parse_args()

    limit = 64 if args.smoke else None
    results_dir = args.results_dir or ("results_smoke" if args.smoke else config.RESULTS_DIR)
    os.makedirs(os.path.join(PROJECT_ROOT, results_dir), exist_ok=True)
    python = sys.executable
    data_args = ["--results_dir", results_dir] + (["--limit", str(limit)] if limit else [])

    preflight(args.methods)

    start = time.time()
    run([python, "-m", "src.environment", "--threads", str(args.threads), "--results_dir", results_dir])
    run([python, "-m", "src.macs"] + data_args)
    for method in args.methods:
        run([python, "-m", "src.benchmark", "--method", method,
             "--threads", str(args.threads)] + data_args)
    run([python, "-m", "src.summarize", "--results_dir", results_dir])
    print(f"\nAll done in {(time.time() - start) / 60:.1f} min. Results are in '{results_dir}/'.")


if __name__ == "__main__":
    main()
