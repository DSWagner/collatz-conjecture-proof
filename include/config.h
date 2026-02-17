// =============================================================================
// config.h - Collatz Conjecture Proof Assistant v1.0.0
// =============================================================================
// GPU-accelerated verification of 4 mathematical directions toward proof:
//   D1: Cycle impossibility via exact residue class enumeration
//   D2: Characterize delayed & near-cycle numbers (binary structure analysis)
//   D3: Stopping time distribution vs Terras theorem
//   D4: Drift bound -- exact expected log-ratio per residue class mod 2^k
// =============================================================================

#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>

// GPU configuration (RTX 3070)
constexpr int BLOCK_SIZE      = 256;
constexpr int NUM_SMS         = 46;
constexpr int BLOCKS_PER_SM   = 6;
constexpr int GRID_SIZE       = NUM_SMS * BLOCKS_PER_SM; // 276

// Batch sizes
constexpr uint64_t BATCH_SIZE_MAIN = 16ULL << 20; // 16M

// D1: Cycle analysis
// Enumerate k odd steps, L = ceil(k * log2(3)) divisions.
// For k <= 64, 3^k fits in ~102 bits -- use double for range check.
constexpr int MAX_K_CYCLE = 64;

// D2: Near-cycle & delayed number extraction
constexpr int MAX_NEAR_CYCLE_STORED  = 5000;
constexpr int MAX_DELAYED_STORED     = 5000;
constexpr float NEAR_CYCLE_THRESH    = 0.95f;
constexpr float DELAYED_FACTOR       = 10.0f;

// D3: Stopping time distribution
constexpr int STOP_HIST_BINS  = 500;

// D4: Drift bound by residue class mod 2^k
constexpr int MAX_DRIFT_K = 24;

// Theoretical constants
constexpr double LOG2_3             = 1.5849625007211563;
constexpr double THEORETICAL_RATIO  = 0.6309297535714574; // log(2)/log(3)
// log2(3) - 2 = expected drift per Syracuse step
constexpr double LOG2_3_MINUS_2     = -0.41503749927884376;

// Default run sizes
constexpr uint64_t DEFAULT_D2_COUNT = 100000000ULL;   // 100M
constexpr uint64_t DEFAULT_D3_COUNT = 1000000000ULL;  // 1B
constexpr uint32_t DEFAULT_MAX_STEPS = 100000;

#endif // CONFIG_H
