import argparse
from pathlib import Path
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Random Matrix Ensembles
# =========================

def goe_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate an n x n real symmetric matrix from a GOE-like ensemble.
    """
    A = rng.standard_normal(size=(n, n))
    H = (A + A.T) / 2
    return H / np.sqrt(n)


def gue_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate an n x n complex Hermitian matrix from a GUE-like ensemble.
    """
    A = rng.standard_normal(size=(n, n)) + 1j * rng.standard_normal(size=(n, n))
    H = (A + A.conj().T) / 2
    return H / np.sqrt(n)


def poisson_eigenvalues(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate uncorrelated eigenvalues as a Poisson baseline.
    """
    return np.sort(rng.uniform(-1, 1, size=n))


# =========================
# Spacing Computation
# =========================

def normalized_spacings(eigs: np.ndarray, trim_fraction: float = 0.15) -> np.ndarray:
    """
    Sort eigenvalues, remove edge eigenvalues, compute neighboring gaps,
    then normalize gaps so the average spacing is 1.
    """
    eigs = np.sort(eigs.real)
    n = len(eigs)

    if not 0 <= trim_fraction < 0.5:
        raise ValueError("--trim-fraction must be in the range [0, 0.5).")

    lo = int(trim_fraction * n)
    hi = int((1 - trim_fraction) * n)
    center = eigs[lo:hi]

    if len(center) < 2:
        raise ValueError(
            "Matrix is too small after trimming. "
            "Use a larger matrix or reduce --trim-fraction."
        )

    gaps = np.diff(center)
    mean_gap = np.mean(gaps)
    if mean_gap <= 0:
        raise ValueError("Mean spacing is zero or negative; cannot normalize spacings.")

    return gaps / mean_gap


def sample_eigenvalues(ensemble: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate eigenvalues from the selected ensemble.
    """
    if ensemble == "goe":
        H = goe_matrix(n, rng)
        return np.linalg.eigvalsh(H)
    if ensemble == "gue":
        H = gue_matrix(n, rng)
        return np.linalg.eigvalsh(H)
    if ensemble == "poisson":
        return poisson_eigenvalues(n, rng)
    raise ValueError(f"Unknown ensemble: {ensemble}")


def _worker_run_chunk(args: tuple) -> np.ndarray:
    """
    Worker function for multiprocessing. Runs a chunk of Monte Carlo trials
    using its own seeded random generator for reproducibility.

    Each worker receives a SeedSequence spawned from the master, which
    guarantees statistically independent streams across processes.
    """
    ensemble, n, trials_in_chunk, trim_fraction, seed_seq = args
    rng = np.random.default_rng(seed_seq)
    chunk = []
    for _ in range(trials_in_chunk):
        eigs = sample_eigenvalues(ensemble, n, rng)
        chunk.append(normalized_spacings(eigs, trim_fraction=trim_fraction))
    return np.concatenate(chunk) if chunk else np.empty(0)


def collect_spacings(
    ensemble: str,
    n: int,
    trials: int,
    trim_fraction: float,
    workers: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """
    Run Monte Carlo trials and collect normalized eigenvalue spacings.

    Parameters
    ----------
    workers : int
        Number of parallel processes. workers=1 runs serially.
    seed : int | None
        Master seed. Each worker is given an independent stream derived
        from this seed using NumPy's SeedSequence.spawn(), so results are
        reproducible regardless of worker count.
    """
    seed_seq = np.random.SeedSequence(seed)

    if workers <= 1:
        rng = np.random.default_rng(seed_seq)
        all_spacings = []
        for _ in range(trials):
            eigs = sample_eigenvalues(ensemble, n, rng)
            all_spacings.append(normalized_spacings(eigs, trim_fraction=trim_fraction))
        return np.concatenate(all_spacings)

    base, extra = divmod(trials, workers)
    chunk_sizes = [base + (1 if i < extra else 0) for i in range(workers)]
    chunk_sizes = [c for c in chunk_sizes if c > 0]

    child_seeds = seed_seq.spawn(len(chunk_sizes))
    job_args = [
        (ensemble, n, size, trim_fraction, child_seeds[i])
        for i, size in enumerate(chunk_sizes)
    ]

    with Pool(processes=len(chunk_sizes)) as pool:
        results = pool.map(_worker_run_chunk, job_args)

    return np.concatenate(results)


# =========================
# Input Matrix Loading
# =========================

def load_matrices_from_file(path: str) -> list[np.ndarray]:
    """
    Load one or more square matrices from a file.
    Supported formats: .npy, .npz, .txt, .csv, .dat
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".npy":
        data = np.load(path_obj, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path_obj, allow_pickle=False)
        if len(archive.files) != 1:
            raise ValueError(".npz file must contain exactly one array.")
        data = archive[archive.files[0]]
    elif suffix in {".txt", ".csv", ".dat"}:
        delimiter = "," if suffix == ".csv" else None
        data = np.loadtxt(path_obj, dtype=np.complex128, delimiter=delimiter)
    else:
        raise ValueError("Unsupported file type. Use .npy, .npz, .txt, .csv, or .dat.")

    if data.ndim == 2:
        matrices = [data]
    elif data.ndim == 3:
        matrices = [data[i] for i in range(data.shape[0])]
    else:
        raise ValueError("Input must contain a square matrix or a stack of square matrices.")

    for i, matrix in enumerate(matrices):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Matrix {i} is not square.")
        if not np.allclose(matrix, matrix.conj().T, atol=1e-8):
            warnings.warn(
                f"Matrix {i} is not Hermitian; symmetrizing using (M + M^H) / 2.",
                UserWarning,
            )
            matrices[i] = (matrix + matrix.conj().T) / 2

    return matrices


def spacings_from_matrices(
    matrices: list[np.ndarray],
    trim_fraction: float,
) -> np.ndarray:
    """
    Compute normalized spacings from input matrices.
    """
    all_spacings = []
    for matrix in matrices:
        eigs = np.linalg.eigvalsh(matrix)
        spacings = normalized_spacings(eigs, trim_fraction=trim_fraction)
        all_spacings.append(spacings)
    return np.concatenate(all_spacings)


# =========================
# Theory Curves
# =========================

def theory_curves(s: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return theoretical nearest-neighbor spacing curves.
    """
    return {
        "GOE Wigner surmise": (np.pi / 2) * s * np.exp(-np.pi * s**2 / 4),
        "GUE Wigner surmise": (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi),
        "Poisson baseline": np.exp(-s),
    }


# =========================
# Plotting
# =========================

def plot_results(
    spacings: np.ndarray,
    bins: int,
    title: str,
    selected_ensemble: str | None = None,
) -> None:
    """
    Plot empirical spacing histogram with theoretical curves.
    """
    s = np.linspace(0, 4, 400)
    curves = theory_curves(s)

    plt.figure(figsize=(9, 5.5))
    plt.hist(
        spacings,
        bins=bins,
        density=True,
        alpha=0.65,
        label="Empirical spacings",
    )

    for name, curve in curves.items():
        linewidth = 2.7 if selected_ensemble and selected_ensemble.upper() in name else 1.8
        plt.plot(s, curve, linewidth=linewidth, label=name)

    plt.xlabel("Normalized nearest-neighbor spacing")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_input_vs_simulated(
    input_spacings: np.ndarray,
    goe_spacings: np.ndarray,
    gue_spacings: np.ndarray,
    poisson_spacings: np.ndarray,
    bins: int,
) -> None:
    """
    Compare input matrix spacings against simulated GOE, GUE, and Poisson spacings,
    with theoretical curves overlaid.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(input_spacings, bins=bins, density=True, alpha=0.55, label="Input matrix")
    plt.hist(goe_spacings, bins=bins, density=True, alpha=0.30, label="Simulated GOE")
    plt.hist(gue_spacings, bins=bins, density=True, alpha=0.30, label="Simulated GUE")
    plt.hist(poisson_spacings, bins=bins, density=True, alpha=0.30, label="Simulated Poisson")

    s = np.linspace(0, 4, 400)
    curves = theory_curves(s)
    plt.plot(s, curves["GOE Wigner surmise"], linewidth=1.5, linestyle="--", label="GOE theory")
    plt.plot(s, curves["GUE Wigner surmise"], linewidth=1.5, linestyle="--", label="GUE theory")
    plt.plot(s, curves["Poisson baseline"], linewidth=1.5, linestyle="--", label="Poisson theory")

    plt.xlabel("Normalized nearest-neighbor spacing")
    plt.ylabel("Density")
    plt.title("Input Matrix vs Simulated Ensembles vs Theory")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =========================
# Reporting
# =========================

def print_summary(spacings: np.ndarray, label: str = "Dataset") -> None:
    print(f"\n===== Summary Statistics: {label} =====")
    print(f"Collected spacings: {len(spacings)}")
    print(f"Mean spacing: {np.mean(spacings):.4f}")
    print(f"Median spacing: {np.median(spacings):.4f}")
    print(f"Minimum spacing: {np.min(spacings):.4f}")
    print(f"Standard deviation: {np.std(spacings):.4f}")
    print(f"P(s < 0.1): {np.mean(spacings < 0.1):.4f}")
    print(f"P(s < 0.2): {np.mean(spacings < 0.2):.4f}")


def print_interpretation() -> None:
    print("\nInterpretation:")
    print("- Smaller P(s < 0.1) suggests stronger level repulsion.")
    print("- GOE/GUE ensembles should suppress tiny gaps more than Poisson.")
    print("- Poisson spacings represent an uncorrelated baseline with no repulsion.")


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo spectral simulation of random matrix ensembles. "
            "Generates GOE, GUE, or Poisson spectra and compares empirical "
            "nearest-neighbor spacings against theoretical curves."
        )
    )
    parser.add_argument("--ensemble", choices=["goe", "gue", "poisson"], default="gue")
    parser.add_argument("--input", "-i", type=str, help="Path to a matrix file.")
    parser.add_argument("--compare-simulated", action="store_true")
    parser.add_argument("--n", type=int, default=120, help="Matrix dimension.")
    parser.add_argument("--trials", type=int, default=80, help="Number of Monte Carlo trials.")
    parser.add_argument("--trim-fraction", type=float, default=0.15)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--seed", type=int, default=281)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for Monte Carlo trials. "
            "Use 1 (default) for serial execution, or -1 to use all available CPU cores."
        ),
    )
    args = parser.parse_args()

    if args.n < 3:
        raise ValueError("--n must be at least 3.")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")

    workers = cpu_count() if args.workers == -1 else max(1, args.workers)

    if args.input:
        matrices = load_matrices_from_file(args.input)
        spacings = spacings_from_matrices(matrices, trim_fraction=args.trim_fraction)
        print(f"Loaded {len(matrices)} matrix/matrices from {args.input}.")
        print_summary(spacings, label="Input Matrix")
        print_interpretation()

        if args.compare_simulated:
            goe_spacings = collect_spacings(
                "goe", args.n, args.trials, args.trim_fraction, workers, args.seed
            )
            gue_spacings = collect_spacings(
                "gue", args.n, args.trials, args.trim_fraction, workers, args.seed + 1
            )
            poisson_spacings = collect_spacings(
                "poisson", args.n, args.trials, args.trim_fraction, workers, args.seed + 2
            )
            print_summary(goe_spacings, label="Simulated GOE")
            print_summary(gue_spacings, label="Simulated GUE")
            print_summary(poisson_spacings, label="Simulated Poisson")
            print_interpretation()

            if not args.no_plot:
                plot_input_vs_simulated(
                    spacings, goe_spacings, gue_spacings, poisson_spacings, args.bins
                )
        elif not args.no_plot:
            plot_results(
                spacings, args.bins,
                "Spectral Spacing Analysis of Input Matrix Data",
                selected_ensemble=None,
            )
    else:
        spacings = collect_spacings(
            args.ensemble, args.n, args.trials, args.trim_fraction, workers, args.seed
        )
        worker_msg = f" using {workers} worker(s)" if workers > 1 else ""
        print(
            f"Generated {args.trials} Monte Carlo samples from "
            f"{args.ensemble.upper()} with n = {args.n}{worker_msg}."
        )
        print_summary(spacings, label=args.ensemble.upper())
        print_interpretation()

        if not args.no_plot:
            plot_results(
                spacings, args.bins,
                f"Monte Carlo Spectral Simulation: {args.ensemble.upper()} Ensemble",
                selected_ensemble=args.ensemble,
            )


if __name__ == "__main__":
    import multiprocessing
    import sys
    # Use 'fork' on Unix/macOS for fast worker startup; 'spawn' is the only
    # option on Windows. Wrap in try/except in case the start method has
    # already been set elsewhere.
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("fork", force=True)
        except RuntimeError:
            pass
    main()