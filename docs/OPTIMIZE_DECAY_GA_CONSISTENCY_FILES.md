# Understanding the `optimize_decay_ga_consistency` Files

This document explains the multiple versions of decay parameter optimization scripts in the repository.

## Background

These scripts are experimental implementations for optimizing Elo rating decay parameters using genetic algorithms. They all use DEAP (the genetic algorithm framework) and focus on finding parameters that work **consistently** across different time periods, rather than just maximizing performance on a single dataset.

See `CONSISTENCY_OPTIMIZATION_APPROACH.md` for the research methodology behind these experiments.

## The Files Explained

### Core Concept
All these files optimize 4 decay parameters:
1. **quick_succession_days** (20-120): Days threshold for quick succession bonus
2. **quick_succession_bump** (1.01-1.15): Rating multiplier for fighters who compete frequently
3. **decay_days** (180-720): Days before ratings start decaying
4. **decay_rate** (0.0001-0.01): Rate at which ratings decay over time

### File Breakdown

#### 1. `optimize_decay_ga_consistency.py` (505 lines)
**Purpose**: Original consistency-focused optimization  
**Fitness Function**: Composite metric combining:
- Mean ROI across validation windows
- Sharpe ratio (mean/std of ROI)
- Calibration quality (ECE, Brier score)
- Trend stability

**Status**: Base implementation with multiple fitness components

---

#### 2. `optimize_decay_ga_consistency_FINAL.py` (408 lines) ⭐
**Purpose**: Sharpe ratio optimization (recommended approach)  
**Fitness Function**: `Sharpe_ratio = mean_ROI / std_ROI` across rolling validation windows  
**Key Features**:
- Splits validation into quarterly windows
- Optimizes for consistent performance (low volatility)
- Clean implementation with parameter bounds checking
- Mentioned in `CONSISTENCY_OPTIMIZATION_APPROACH.md` as the primary script

**Status**: ✅ Production-ready, documented approach

---

#### 3. `optimize_decay_ga_consistency_WORKING.py` (450 lines)
**Purpose**: Alternative approach using Brier score consistency  
**Fitness Function**: `-mean_brier / std_brier` (Sharpe-like for calibration)  
**Key Features**:
- Optimizes for low and consistent Brier scores
- Focus on probability calibration rather than ROI
- Similar structure to FINAL but different optimization target

**Status**: Working alternative for calibration-focused optimization

---

#### 4. `optimize_decay_ga_consistency_final.py` (367 lines)
**Purpose**: Streamlined Sharpe ratio optimization  
**Fitness Function**: Sharpe ratio with simpler implementation  
**Key Features**:
- Cleaner, more minimal code
- Same concept as FINAL but with fewer validation checks
- Earlier version before robust error handling was added

**Status**: Working but superseded by FINAL

---

#### 5. `optimize_decay_ga_consistency_final_v2.py` (407 lines)
**Purpose**: Variant of FINAL with modifications  
**Fitness Function**: Sharpe ratio (similar to FINAL)  
**Status**: Experimental iteration

---

#### 6. `optimize_decay_ga_consistency_robust.py` (400 lines)
**Purpose**: Enhanced error handling version  
**Fitness Function**: Sharpe ratio with robust error handling  
**Key Features**:
- Additional numerical stability checks
- Better handling of edge cases
- More defensive programming

**Status**: Experimental robustness improvements

---

#### 7. `optimize_decay_ga_consistency_v2.py` (402 lines)
**Purpose**: Earlier version with basic GA implementation  
**Docstring**: "Genetic Algorithm-based decay parameter optimization"  
**Status**: ⚠️ Legacy/deprecated - superseded by consistency-focused versions

---

#### 8. `optimize_decay_ga_consistency_working.py` (402 lines)
**Purpose**: Development/testing version  
**Status**: ⚠️ Likely a working copy during development

---

## Which One Should You Use?

### For Production Use:
**Use `optimize_decay_ga_consistency_FINAL.py`** ⭐

This is the primary implementation mentioned in the research documentation. It:
- Has clean, well-tested code
- Uses the Sharpe ratio approach for consistency
- Is properly documented in `CONSISTENCY_OPTIMIZATION_APPROACH.md`
- Has robust parameter bounds checking

### For Experimentation:
- **`optimize_decay_ga_consistency_WORKING.py`**: If you want to optimize for calibration (Brier score) instead of ROI
- **`optimize_decay_ga_consistency.py`**: If you want a composite fitness function with multiple metrics

## Recommendations for Cleanup

To reduce confusion, consider:

1. **Keep**:
   - `optimize_decay_ga_consistency_FINAL.py` (primary production script)
   - `optimize_decay_ga_consistency_WORKING.py` (calibration alternative)
   - `optimize_decay_ga_consistency.py` (composite fitness baseline)

2. **Archive or Remove**:
   - `optimize_decay_ga_consistency_final.py` (superseded by FINAL)
   - `optimize_decay_ga_consistency_final_v2.py` (experimental iteration)
   - `optimize_decay_ga_consistency_v2.py` (legacy version)
   - `optimize_decay_ga_consistency_working.py` (development copy)
   - `optimize_decay_ga_consistency_robust.py` (experimental variant)

3. **Add to Documentation**:
   - Update `CONSISTENCY_OPTIMIZATION_APPROACH.md` to mention which script to use
   - Add this file to the docs/ directory for future reference

## Running the Scripts

All scripts follow a similar pattern:

```bash
# Primary recommended approach
python optimize_decay_ga_consistency_FINAL.py

# For calibration-focused optimization
python optimize_decay_ga_consistency_WORKING.py
```

They will:
1. Load data from `data/interleaved_cleaned.csv`
2. Split into training/validation (never touching OOS)
3. Run genetic algorithm optimization
4. Output best parameters and validation performance
5. Perform final blind test on OOS data (if available)

## Key Differences Summary

| File | Primary Metric | Focus | Status |
|------|---------------|-------|--------|
| `_FINAL.py` | Sharpe ratio (ROI) | Profit consistency | ✅ Production |
| `_WORKING.py` | Sharpe ratio (Brier) | Calibration consistency | ✅ Alternative |
| `.py` (base) | Composite | Multiple objectives | ✅ Research |
| `_final.py` | Sharpe ratio (ROI) | Simpler version | ⚠️ Superseded |
| `_final_v2.py` | Sharpe ratio | Experimental | ⚠️ Experimental |
| `_robust.py` | Sharpe ratio | Error handling | ⚠️ Experimental |
| `_v2.py` | Basic | Legacy | ⚠️ Deprecated |
| `_working.py` | Unknown | Dev copy | ⚠️ Unclear |

