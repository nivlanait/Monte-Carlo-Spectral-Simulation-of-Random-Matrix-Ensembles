# Monte Carlo Spectral Simulation of Random Matrix Ensembles

This project implements a Monte Carlo simulation framework to study eigenvalue spacing statistics in random matrix ensembles. It demonstrates level repulsion and spectral universality by comparing empirical results with theoretical predictions from Random Matrix Theory.

---

## 📌 Overview

Random Matrix Theory predicts that eigenvalues of large random matrices are not independent. Instead, they exhibit **level repulsion**, meaning nearby eigenvalues avoid being too close together.

This project:

- Generates random matrices from GOE and GUE ensembles
- Computes eigenvalues and nearest-neighbor spacings
- Normalizes spacings for statistical comparison
- Simulates Poisson-distributed eigenvalues as a baseline
- Compares empirical distributions against theoretical models
- Allows analysis of user-provided matrices

---

## 🧠 Key Concepts

### Eigenvalue Spacing

For sorted eigenvalues:

λ₁ ≤ λ₂ ≤ ... ≤ λₙ

Spacing is defined as:

sᵢ = λᵢ₊₁ − λᵢ

---

### Level Repulsion

- GOE: linear suppression near zero  
- GUE: stronger (quadratic) suppression  
- Poisson: no suppression  

---

### Theoretical Distributions

GOE (Wigner surmise):

P(s) = (π/2) s e^(−π s² / 4)

GUE:

P(s) = (32 / π²) s² e^(−4s² / π)

Poisson:

P(s) = e^(−s)

---

## ⚙️ Methodology

### Monte Carlo Simulation

For each trial:

1. Generate a random matrix (GOE/GUE)
2. Compute eigenvalues
3. Calculate spacings
4. Normalize spacings
5. Aggregate across trials

---

### Input Matrix Mode

You can also analyze real matrices:

- Loads matrix from file
- Computes eigenvalues
- Analyzes spacing distribution
- Optionally compares to simulated ensembles

---

## 📊 Features

- GOE / GUE / Poisson simulation
- Input matrix analysis (`--input`)
- Comparison mode (`--compare-simulated`)
- Theoretical curve overlay
- Statistical summary (mean, variance, small-gap probability)

---

## 🚀 Usage

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib

---

## ⚡ Parallel Execution & Benchmarks

The Monte Carlo loop is embarrassingly parallel — each trial generates an
independent matrix and computes its eigenvalues. The simulation supports
multi-process parallelism via Python's `multiprocessing` module.

### Usage

Run on all available CPU cores:

    python main.py --ensemble gue --n 400 --trials 1000 --workers -1

Or specify an explicit worker count:

    python main.py --ensemble goe --n 200 --trials 500 --workers 4

### Reproducibility

Each worker is seeded via `numpy.random.SeedSequence.spawn()`, which
produces statistically independent random streams across processes.
Results are reproducible for a fixed `(seed, workers)` pair, and the
empirical spacing distribution converges to the same Wigner surmise
regardless of worker count.

### Benchmarks

Run the included benchmark to measure speedup on your hardware. Because
NumPy's eigendecomposition routines internally use multi-threaded BLAS,
the benchmark must be run with single-threaded BLAS to avoid thread
oversubscription between worker processes:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        python benchmark.py --sizes 100 200 400 --trials 200

Sample output on a 12-core machine (GUE ensemble, 200 trials per config,
best-of-3):

| Matrix size | Workers | Wall time (s) | Speedup | Efficiency |
|:-----------:|:-------:|:-------------:|:-------:|:----------:|
|   n = 100   |    1    |     0.177     |  1.00x  |    100%    |
|   n = 100   |    2    |     0.128     |  1.39x  |     69%    |
|   n = 100   |    4    |     0.108     |  1.64x  |     41%    |
|   n = 100   |    8    |     0.098     |  1.82x  |     23%    |
|   n = 100   |   12    |     0.097     |  1.82x  |     15%    |
|   n = 200   |    1    |     1.149     |  1.00x  |    100%    |
|   n = 200   |    2    |     0.843     |  1.36x  |     68%    |
|   n = 200   |    4    |     0.623     |  1.84x  |     46%    |
|   n = 200   |    8    |     0.469     |  2.45x  |     31%    |
|   n = 200   |   12    |     0.515     |  2.23x  |     19%    |
|   n = 400   |    1    |     6.710     |  1.00x  |    100%    |
|   n = 400   |    2    |     4.311     |  1.56x  |     78%    |
|   n = 400   |    4    |     3.321     |  2.02x  |     51%    |
|   n = 400   |    8    |     2.800     |  2.40x  |     30%    |
|   n = 400   |   12    |     2.557     |  2.62x  |     22%    |

### Interpretation

Speedup grows with matrix size — from a peak of 1.82x at n=100 to 2.62x
at n=400 — because the per-trial eigendecomposition cost (O(n³)) grows
faster than the fixed inter-process communication overhead. For small
matrices, worker startup and pickling dominate, so adding processes
yields diminishing returns.

Throughput plateaus beyond 8 workers on this 12-core machine, which is
consistent with 6 physical cores plus simultaneous multithreading (SMT):
hyperthreads share execution units and contribute less than physical
cores on math-bound workloads like dense linear algebra.

Without setting `OMP_NUM_THREADS=1`, parallel runs are *slower* than
serial because each worker process spawns its own thread pool inside
NumPy's BLAS, leading to thread oversubscription and contention. This
is a common pitfall when combining process-level and thread-level
parallelism in scientific Python.

The benchmark script also produces `benchmark_speedup.png`, plotting
speedup against worker count for each matrix size, with an ideal-linear
reference line.