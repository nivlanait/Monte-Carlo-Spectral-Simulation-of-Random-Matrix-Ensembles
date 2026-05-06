"""
Benchmark: serial vs parallel Monte Carlo simulation of random matrix ensembles.

Measures wall-clock time for collect_spacings() across a range of matrix sizes
and worker counts, prints a comparison table, and saves a speedup plot.

Usage:
    python benchmark.py
    python benchmark.py --sizes 100 200 400 --trials 200
    python benchmark.py --max-workers 12 --output bench.png
"""

import argparse
import time
from multiprocessing import cpu_count

import numpy as np
import matplotlib.pyplot as plt

from main import collect_spacings


def time_run(
    ensemble: str,
    n: int,
    trials: int,
    workers: int,
    trim_fraction: float,
    seed: int,
    repeats: int = 3,
) -> tuple[float, float]:
    """
    Run collect_spacings several times and return (best_time, median_time).
    Uses best-of-K for the headline number (least noisy on a busy laptop)
    and median for context.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = collect_spacings(
            ensemble=ensemble,
            n=n,
            trials=trials,
            trim_fraction=trim_fraction,
            workers=workers,
            seed=seed,
        )
        times.append(time.perf_counter() - t0)
    return min(times), float(np.median(times))


def run_benchmark(
    ensemble: str,
    sizes: list[int],
    trials: int,
    worker_counts: list[int],
    trim_fraction: float,
    seed: int,
    repeats: int,
) -> dict:
    """
    Returns a dict mapping n -> {workers: (best_time, median_time)}.
    """
    results: dict[int, dict[int, tuple[float, float]]] = {}

    print(f"\nBenchmark: ensemble={ensemble.upper()}, trials={trials}, "
          f"repeats={repeats} per config")
    print("=" * 72)

    for n in sizes:
        results[n] = {}
        print(f"\nMatrix size n = {n}")
        print("-" * 72)
        print(f"  {'Workers':>8}  {'Best (s)':>10}  {'Median (s)':>12}  "
              f"{'Speedup':>9}  {'Efficiency':>11}")

        baseline = None
        for w in worker_counts:
            best, med = time_run(ensemble, n, trials, w, trim_fraction, seed, repeats)
            results[n][w] = (best, med)
            if w == 1:
                baseline = best
            speedup = baseline / best if baseline else 1.0
            efficiency = speedup / w
            print(f"  {w:>8}  {best:>10.3f}  {med:>12.3f}  "
                  f"{speedup:>8.2f}x  {efficiency:>10.1%}")

    return results


def plot_speedup(results: dict, output_path: str, ensemble: str) -> None:
    """
    Plot speedup vs worker count for each matrix size, plus the ideal line.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    all_workers = sorted({w for r in results.values() for w in r})
    ax.plot(all_workers, all_workers, "k--", alpha=0.4, label="Ideal (linear)")

    for n, runs in results.items():
        workers = sorted(runs.keys())
        baseline = runs[1][0]
        speedups = [baseline / runs[w][0] for w in workers]
        ax.plot(workers, speedups, marker="o", label=f"n = {n}")

    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("Speedup (vs. serial)")
    ax.set_title(f"Parallel Monte Carlo Speedup — {ensemble.upper()} Ensemble")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(all_workers)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    print(f"\nSaved speedup plot to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs parallel Monte Carlo random matrix simulation."
    )
    parser.add_argument("--ensemble", choices=["goe", "gue", "poisson"], default="gue")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Matrix dimensions to benchmark.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=200,
        help="Monte Carlo trials per configuration.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum worker count (default: all available cores).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to run each config (best-of-K reported).",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--output", type=str, default="benchmark_speedup.png")
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting (table only)."
    )
    args = parser.parse_args()

    max_w = args.max_workers if args.max_workers else cpu_count()
    worker_counts = sorted({1, 2, 4, max_w} | {w for w in [2, 4, 8] if w <= max_w})
    worker_counts = [w for w in worker_counts if w <= max_w]

    print(f"Detected {cpu_count()} CPU cores. Testing worker counts: {worker_counts}")

    results = run_benchmark(
        ensemble=args.ensemble,
        sizes=args.sizes,
        trials=args.trials,
        worker_counts=worker_counts,
        trim_fraction=args.trim_fraction,
        seed=args.seed,
        repeats=args.repeats,
    )

    if not args.no_plot:
        plot_speedup(results, args.output, args.ensemble)


if __name__ == "__main__":
    main()