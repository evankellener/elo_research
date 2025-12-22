# Understanding the `optimize_decay_ga_consistency` Files

This document explains the decay parameter optimization scripts in the repository.

## ⭐ NEW: Unified Script (December 2024)

**Use `optimize_decay_ga_consistency.py` with command-line flags**

The main optimization script has been unified to support multiple optimization modes through command-line arguments:

```bash
# Show all available options
python optimize_decay_ga_consistency.py --help

# Run with different optimization modes:
python optimize_decay_ga_consistency.py --mode combined    # Default: Sharpe + ROI + trend + calibration
python optimize_decay_ga_consistency.py --mode sharpe      # Pure Sharpe ratio optimization
python optimize_decay_ga_consistency.py --mode roi         # Direct ROI optimization
python optimize_decay_ga_consistency.py --mode brier       # Brier score-based (calibration-focused)

# Customize GA parameters:
python optimize_decay_ga_consistency.py --mode sharpe --population 30 --generations 20
python optimize_decay_ga_consistency.py --mode roi --output custom_results.json
```

### Available Optimization Modes

1. **`combined`** (default): Composite fitness with Sharpe ratio, mean ROI, trend stability, and calibration
2. **`sharpe`**: Pure Sharpe ratio optimization for consistent performance
3. **`roi`**: Direct mean ROI optimization
4. **`brier`**: Brier score-based optimization for probability calibration

### Command-Line Options

- `--mode {combined,sharpe,roi,brier}`: Optimization mode (default: combined)
- `--population N`: Population size for GA (default: 20)
- `--generations N`: Number of generations (default: 15)
- `--output FILE`: Output JSON file (default: `decay_ga_{mode}_results.json`)

### Fixes in Latest Version

- **Fixed extremely high fitness values**: Added epsilon check to prevent division by near-zero in Sharpe ratio calculation
- **Consolidated multiple variants**: All optimization modes now available in a single script with flags

## Background

These scripts optimize Elo rating decay parameters using genetic algorithms. They focus on finding parameters that work **consistently** across different time periods, rather than just maximizing performance on a single dataset.

See `CONSISTENCY_OPTIMIZATION_APPROACH.md` for the research methodology behind these experiments.

## The Parameters Being Optimized

All optimization modes optimize 4 decay parameters:
1. **quick_succession_days** (20-120): Days threshold for quick succession bonus
2. **quick_succession_bump** (1.01-1.15): Rating multiplier for fighters who compete frequently
3. **decay_days** (180-720): Days before ratings start decaying
4. **decay_rate** (0.0001-0.01): Rate at which ratings decay over time

## Legacy Files (Historical Reference)

The following files are older variants that have been consolidated into the unified script:

### `optimize_decay_ga_consistency_FINAL.py` (408 lines)
**Purpose**: Sharpe ratio optimization  
**Fitness Function**: `Sharpe_ratio = mean_ROI / std_ROI`  
**Status**: ⚠️ Superseded by unified script with `--mode sharpe`

### `optimize_decay_ga_consistency_WORKING.py` (450 lines)
**Purpose**: Brier score consistency optimization  
**Fitness Function**: `-mean_brier / std_brier`  
**Status**: ⚠️ Superseded by unified script with `--mode brier`

### Other Variants
- `optimize_decay_ga_consistency_final.py` - Earlier Sharpe implementation
- `optimize_decay_ga_consistency_final_v2.py` - Experimental iteration
- `optimize_decay_ga_consistency_robust.py` - Enhanced error handling
- `optimize_decay_ga_consistency_v2.py` - Legacy basic GA
- `optimize_decay_ga_consistency_working.py` - Development copy

**Recommendation**: These legacy files can be archived or removed as their functionality is now available in the unified script.

## Migration Guide

If you were using an old script, here's how to migrate to the unified version:

| Old Script | New Command |
|------------|-------------|
| `optimize_decay_ga_consistency_FINAL.py` | `python optimize_decay_ga_consistency.py --mode sharpe` |
| `optimize_decay_ga_consistency_WORKING.py` | `python optimize_decay_ga_consistency.py --mode brier` |
| Original base script with composite | `python optimize_decay_ga_consistency.py --mode combined` |

## Output

All modes produce a JSON results file containing:
- Best parameters found
- Fitness score
- Validation performance metrics (ROI, accuracy, ECE, Brier)
- Out-of-sample performance (if available)
- Generation history

Default output files: `decay_ga_{mode}_results.json`

## Examples

```bash
# Quick test with fewer generations
python optimize_decay_ga_consistency.py --mode sharpe --generations 5 --population 10

# Full optimization with custom output
python optimize_decay_ga_consistency.py --mode combined --generations 20 --population 30 --output full_optimization.json

# Calibration-focused optimization
python optimize_decay_ga_consistency.py --mode brier --output calibration_results.json
```

## Troubleshooting

### Extremely High Fitness Values
This issue has been fixed in the latest version. The Sharpe ratio calculation now includes an epsilon check to prevent division by near-zero standard deviations.

### Missing Data Files
Ensure `data/interleaved_cleaned.csv` exists in your repository. For out-of-sample evaluation, `data/past3_events.csv` should also be available.

### Long Runtime
Reduce `--generations` and `--population` for faster testing. Default values (15 generations, 20 population) provide a good balance.
