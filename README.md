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