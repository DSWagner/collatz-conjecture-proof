# Collatz Conjecture Proof Assistant

GPU-accelerated CUDA program testing 4 mathematical directions toward a proof of the Collatz conjecture.

## Hardware
- GPU: NVIDIA RTX 3070 (Ampere, SM 8.6, 46 SMs, 5888 CUDA cores)
- CPU: AMD Ryzen 7 5800X
- OS: Windows 11

## 4 Directions

### D1: Cycle Impossibility
Enumerates all (k, L) pairs where k = odd steps and L = even steps in a candidate cycle.
Shows that any cycle in [2, 10^12] would require k >= 26 odd steps.
Combined with Simons & de Weger (2005): no cycle with k <= 68.

### D2: Binary Structure of Delayed Numbers
Analyzes the binary structure (popcount, alternating-bit score, max run of 1-bits)
of "delayed" numbers (steps > 10 * log2(n)) and "near-cycle" numbers (min >= 0.95 * start).
Identifies whether these numbers cluster in specific residue classes mod 16.

### D3: Stopping Time Distribution
Verifies Terras theorem empirically: fraction of numbers with stopping time <= C * log2(n).
Fits an exponential tail to estimate the probability of delayed convergence.

### D4: Drift Bound Per Residue Class mod 2^k
For each odd residue class c mod 2^k (k = 1..24), computes the exact drift of one
Syracuse step: drift = log2((3c+1)/2^v) - log2(c).
Confirms mean drift ~ -0.4150 (= log2(3) - 2) with bounded maximum drift.

## Build

Requires: CUDA 12.9, VS 2022 Build Tools, CMake 4+, Ninja

```bat
build.bat
```

## Run

```bat
build\collatz_proof.exe
build\collatz_proof.exe --d2 100000000 --d3 1000000000
build\collatz_proof.exe --only 4    (D4 only: fast, ~1 second)
build\collatz_proof.exe --only 12   (D1 + D2)
```

## Versioning

- v1.0.0: Initial 4-direction analysis
