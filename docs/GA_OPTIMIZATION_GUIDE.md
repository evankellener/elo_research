# Genetic Algorithm Optimization Guide

This guide provides a comprehensive explanation of the Genetic Algorithm (GA) optimization system for Elo rating parameters, including all CLI commands and usage examples.

> 📋 **Quick Reference**: Looking for a condensed command reference? See [GA_OPTIMIZATION_QUICK_REFERENCE.md](GA_OPTIMIZATION_QUICK_REFERENCE.md)

## Table of Contents

1. [Overview](#overview)
2. [What is Genetic Algorithm Optimization?](#what-is-genetic-algorithm-optimization)
3. [Available Scripts](#available-scripts)
4. [CLI Commands and Arguments](#cli-commands-and-arguments)
5. [Usage Examples](#usage-examples)
6. [Understanding Parameters](#understanding-parameters)
7. [Understanding Fitness Functions](#understanding-fitness-functions)
8. [Interpreting Results](#interpreting-results)
9. [Advanced Configuration](#advanced-configuration)

---

## Overview

The Elo Research project includes three main genetic algorithm optimization scripts:

1. **`full_genetic_with_k_denom_mov.py`** - Complete multi-parameter optimization with composite fitness
2. **`ga_time_split_roi.py`** - ROI-focused optimization with time-series cross-validation
3. **`example_ga_optimization.py`** - Simple example for learning the basics
4. **`example_composite_fitness.py`** - Examples of different fitness weight configurations

These scripts use evolutionary computation to find optimal Elo rating system parameters that maximize prediction accuracy, calibration, or betting profitability.

---

## What is Genetic Algorithm Optimization?

### The Problem

Traditional grid search for parameter optimization faces exponential complexity. For example:
- Optimizing 1 parameter (K-factor) with 50 values: **50 evaluations**
- Optimizing 7 parameters (K, denominator, 5 MOV weights) with 50 values each: **50^7 = 781 billion evaluations** ❌ (computationally infeasible)

### The Solution: Genetic Algorithms

Genetic algorithms use evolutionary principles to efficiently explore large parameter spaces:

1. **Population**: Create a diverse set of candidate solutions (parameter combinations)
2. **Selection**: Better-performing solutions are more likely to "reproduce"
3. **Crossover**: Combine parameters from two parent solutions to create offspring
4. **Mutation**: Randomly adjust parameters to explore new regions
5. **Elitism**: Always preserve the best solutions found so far

**Result**: GA finds optimal solutions with only **~5,000 evaluations** (50 population × 100 generations) ✅

### Key Advantages

- **Multi-parameter optimization**: Simultaneously optimize 3-8+ parameters
- **Efficiency**: Finds good solutions 100,000x faster than grid search
- **Parameter interactions**: Discovers synergies between parameters that grid search misses
- **Adaptive search**: Focuses exploration on promising regions of parameter space
- **Early stopping**: Automatically stops when converged, saving computation time

---

## Available Scripts

### 1. Full Genetic Algorithm Optimization

**File**: `optimization/full_genetic_with_k_denom_mov.py`

**Purpose**: Complete optimization of all Elo parameters using composite fitness function.

**Optimizes**:
- K-factor (10-500)
- Denominator (200-600)
- Confidence threshold for betting (30-150)
- MOV weights: KO/TKO, Submission, Unanimous Decision, Majority Decision, Split Decision

**Fitness Function**: Composite score combining:
- Accuracy (30%)
- Log Loss (20%)
- Brier Score (20%)
- ROI (30%)

**Use when**: You want the best overall parameters balancing accuracy, calibration, and profitability.

---

### 2. ROI-Focused Optimization with Time-Series Validation

**File**: `optimization/ga_time_split_roi.py`

**Purpose**: Optimize specifically for betting profitability using time-series cross-validation.

**Optimizes**:
- K-factor (10-500)
- Denominator (200-600)
- Confidence threshold (30-150)

**Fitness Function**: Average ROI across multiple time-based validation windows with variance penalty.

**Use when**: Your primary goal is betting profitability and you want robust parameters that work across different time periods.

---

### 3. Basic GA Example

**File**: `examples/example_ga_optimization.py`

**Purpose**: Learn how to use the GA optimization system with a simple example.

**Optimizes**: K-factor and denominator only (simplified)

**Use when**: You're learning the system or need a quick optimization.

---

### 4. Composite Fitness Examples

**File**: `examples/example_composite_fitness.py`

**Purpose**: Demonstrates different fitness weight configurations for different goals.

**Shows**:
- Balanced optimization (equal weights)
- Accuracy-focused optimization (emphasize prediction quality)
- Profit-focused optimization (emphasize ROI)

**Use when**: You want to understand how to customize fitness weights for your specific goals.

---

## CLI Commands and Arguments

### 1. Full Genetic Algorithm Optimization

```bash
python optimization/full_genetic_with_k_denom_mov.py
```

**No command-line arguments** - Configuration is done by editing the script's `__main__` section.

**Configuration in script**:
```python
# Parameter bounds
param_bounds = {
    'k': (10, 500),
    'denominator': (200, 600),
    'confidence_threshold': (30, 150),
    'w_ko': (1.0, 2.0),
    'w_sub': (1.0, 2.0),
    'w_udec': (0.8, 1.5),
    'w_mdec': (0.7, 1.3),
    'w_sdec': (0.5, 1.2),
}

# Fitness weights
fitness_weights = {
    'accuracy': 0.3,
    'log_loss': 0.2,
    'brier_score': 0.2,
    'roi': 0.3
}

# GA settings
optimize_elo_parameters_with_ga(
    df,
    param_bounds=param_bounds,
    population_size=50,
    generations=100,
    elite_size=5,
    mutation_rate=0.15,
    crossover_rate=0.8,
    early_stop_generations=15,
    optimize_for="composite",
    fitness_weights=fitness_weights
)
```

**Input files**:
- `data/interleaved_cleaned.csv` - Main training data
- `data/past3_events.csv` - Out-of-sample test data

**Output files**:
- `ga_optimization_history.csv` - Full optimization history
- `images/ga_convergence.png` - Fitness convergence plot
- `images/ga_parameter_evolution.png` - Parameter evolution plot

---

### 2. ROI-Focused Optimization with Time-Series Validation

```bash
python optimization/ga_time_split_roi.py [OPTIONS]
```

**Command-line arguments**:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-file` | string | `data/interleaved_cleaned.csv` | Path to input data CSV file |
| `--split-months` | int | `6` | Number of months for time-based validation splits |
| `--population-size` | int | `40` | GA population size |
| `--generations` | int | `80` | Number of generations to evolve |
| `--output-prefix` | string | `ga_roi` | Prefix for output files |

**Examples**:

```bash
# Use default settings
python optimization/ga_time_split_roi.py

# Customize data file and split window
python optimization/ga_time_split_roi.py --data-file data/my_fights.csv --split-months 12

# Larger population for more thorough search
python optimization/ga_time_split_roi.py --population-size 80 --generations 150

# Custom output location
python optimization/ga_time_split_roi.py --output-prefix results/my_optimization
```

**Input files**:
- Data file specified by `--data-file` argument

**Output files**:
- `{output_prefix}_history.csv` - Optimization history
- `{output_prefix}_best_params.txt` - Best parameters found
- `images/{output_prefix}_results.png` - ROI optimization results plot

---

### 3. Basic GA Example

```bash
python examples/example_ga_optimization.py
```

**No command-line arguments** - Uses last 3000 fights for quick demonstration.

**Input files**:
- `data/interleaved_cleaned.csv`

**Output**: Console output only (no saved files)

---

### 4. Composite Fitness Examples

```bash
python examples/example_composite_fitness.py
```

**No command-line arguments** - Runs balanced optimization example by default.

**Edit script to run other examples**:
```python
# Uncomment to run additional examples:
best2, ga2 = example_accuracy_focused()
best3, ga3 = example_profit_focused()
```

**Input files**:
- `data/interleaved_cleaned.csv`

**Output**: Console output only (no saved files)

---

## Usage Examples

### Example 1: Basic Optimization (Getting Started)

**Goal**: Learn how GA optimization works

```bash
# Navigate to repository root
cd /path/to/elo_research

# Run basic example (fast, ~5 minutes)
python examples/example_ga_optimization.py
```

**What it does**:
- Uses last 3000 fights for quick execution
- Optimizes K-factor (10-500) and denominator (200-600)
- Runs for 20 generations with population of 20
- Shows fitness improvement over generations
- Displays best parameters found

**Expected output**:
```
======================================================================
GENETIC ALGORITHM OPTIMIZATION EXAMPLE
======================================================================
Loading data...
Using 3000 fights for optimization

Running genetic algorithm...
Generation 1/20: Best=0.5823, Avg=0.5645, Worst=0.5234
Generation 2/20: Best=0.5867, Avg=0.5712, Worst=0.5456
...
Generation 20/20: Best=0.6034, Avg=0.5956, Worst=0.5823

======================================================================
RESULTS
======================================================================
Best K-factor: 72.45
Best denominator: 436.78
Best accuracy: 0.6034
```

---

### Example 2: Full Multi-Parameter Optimization

**Goal**: Find optimal parameters for all Elo system components

```bash
python optimization/full_genetic_with_k_denom_mov.py
```

**What it does**:
- Loads all historical fight data
- Optimizes 8 parameters simultaneously:
  - K-factor
  - Denominator
  - Confidence threshold
  - 5 MOV weights (KO, Sub, UD, MD, SD)
- Uses composite fitness function (accuracy + log loss + Brier + ROI)
- Runs for up to 100 generations (early stopping if converged)
- Generates convergence and parameter evolution plots
- Compares results with grid search baseline
- Saves complete optimization history

**Runtime**: 30-60 minutes depending on data size

**Expected output files**:
```
ga_optimization_history.csv              # Full history with fitness per generation
images/ga_convergence.png                 # Fitness improvement visualization
images/ga_parameter_evolution.png         # How each parameter changed
```

**Console output summary**:
```
======================================================================
OPTIMIZATION COMPLETE
======================================================================

Best parameters found:
  k: 156.7234
  denominator: 378.9012
  confidence_threshold: 67.4532
  w_ko: 1.4523
  w_sub: 1.3234
  w_udec: 1.0123
  w_mdec: 0.8945
  w_sdec: 0.7234

Best composite fitness score: 0.623456
(Composite of: accuracy, log loss, Brier score, and ROI)

=== Comparison with Grid Search ===
Grid search (K only): Val=0.5861, OOS=0.6053
GA (all params):      Val=0.6145, OOS=0.6234
Improvement: Val=+0.0284 (+4.85%), OOS=+0.0181 (+2.99%)
```

---

### Example 3: ROI Optimization for Betting

**Goal**: Optimize specifically for betting profitability

```bash
# Standard 6-month validation windows
python optimization/ga_time_split_roi.py

# Shorter windows for faster-changing meta
python optimization/ga_time_split_roi.py --split-months 3

# Longer windows for more stable estimates
python optimization/ga_time_split_roi.py --split-months 12
```

**What it does**:
- Splits data into multiple time-based validation windows
- For each window:
  1. Train Elo on all data before the window
  2. Evaluate ROI on fights in the window
  3. Calculate profit/loss assuming unit bets on confident predictions
- Optimizes for average ROI across all windows
- Penalizes high variance in ROI (prefer stable returns)
- Saves best parameters for betting strategy

**Runtime**: 20-40 minutes depending on split configuration

**Expected output**:
```
======================================================================
ROI OPTIMIZATION COMPLETE
======================================================================

Best parameters:
  k: 134.5623
  denominator: 412.3456
  confidence_threshold: 89.2341

Best ROI: 0.1234

This represents a 12.34% average return per bet
```

**Interpreting ROI**:
- **Positive ROI**: Profitable betting strategy
  - 0.05-0.15 (5-15%): Good, sustainable returns
  - 0.15-0.30 (15-30%): Excellent returns
  - >0.30 (>30%): Exceptional (verify for overfitting)
- **Negative ROI**: Strategy needs refinement or market is efficient
- **ROI near 0**: Break-even, need better calibration

---

### Example 4: Custom Fitness Weight Configurations

**Goal**: Optimize for different objectives by adjusting metric weights

```bash
python examples/example_composite_fitness.py
```

**Scenarios demonstrated**:

#### Balanced Optimization
```python
fitness_weights = {
    'accuracy': 0.25,      # Equal contribution
    'log_loss': 0.25,      # from all metrics
    'brier_score': 0.25,
    'roi': 0.25
}
```
**Use when**: General-purpose optimization, no specific priorities

#### Accuracy-Focused
```python
fitness_weights = {
    'accuracy': 0.40,      # Emphasize correct predictions
    'log_loss': 0.30,      # and good calibration
    'brier_score': 0.30,
    'roi': 0.00            # Don't care about betting profit
}
```
**Use when**: Building a prediction model, ranking fighters, research

#### Profit-Focused
```python
fitness_weights = {
    'accuracy': 0.15,      # Some accuracy needed
    'log_loss': 0.20,      # Calibration important for betting
    'brier_score': 0.15,
    'roi': 0.50            # Maximize profit!
}
```
**Use when**: Building a betting strategy, focusing on money-making

---

## Understanding Parameters

### K-Factor

**Range**: 10-500

**What it controls**: How much ratings change after each fight

- **Low K (10-50)**: Ratings change slowly, stable but slow to adapt
  - Good for: Established fighters, long-term rankings
  - Bad for: New fighters, rapid skill changes

- **Medium K (50-150)**: Balanced adaptation speed
  - Good for: General use, captures both stability and change
  
- **High K (150-500)**: Ratings change quickly, volatile but responsive
  - Good for: Rapidly improving fighters, short-term predictions
  - Bad for: Can overreact to single upsets

**GA typically finds**: K ≈ 70-180 (lower than grid search default of ~250)

---

### Denominator

**Range**: 200-600

**What it controls**: Rating difference sensitivity (standard Elo uses 400)

- **Low denominator (200-350)**: Small rating differences matter more
  - Rating difference of 100 → ~70% win probability
  - Good for: Competitive sports, close matchups common
  
- **Medium denominator (350-450)**: Standard Elo behavior
  - Rating difference of 100 → ~64% win probability
  - Good for: Most applications, balanced sensitivity
  
- **High denominator (450-600)**: Only large rating differences matter
  - Rating difference of 100 → ~58% win probability
  - Good for: Sports with high variance, upsets common

**Formula impact**:
```
P(Fighter1 wins) = 1 / (1 + 10^((Rating2 - Rating1) / denominator))
```

**GA typically finds**: Denominator ≈ 380-450 (slightly higher than standard)

---

### Confidence Threshold

**Range**: 30-150

**What it controls**: Minimum Elo difference to make a bet

- **Low threshold (30-60)**: Bet on most fights
  - More bets, more volume
  - Lower average edge per bet
  - Higher total variance
  
- **Medium threshold (60-100)**: Selective betting
  - Moderate bet volume
  - Better average edge
  - Balanced risk/return
  
- **High threshold (100-150)**: Only bet on clear favorites
  - Few bets, low volume
  - Highest edge per bet
  - Lowest variance, but fewer opportunities

**Relationship to probability**:
```
Threshold 50 ≈ 0.05 probability edge over 50%
Threshold 100 ≈ 0.10 probability edge over 50%
```

**GA typically finds**: Threshold ≈ 60-90 (moderate selectivity)

---

### MOV Weights (Method of Victory)

**Ranges**:
- `w_ko`: 1.0-2.0 (KO/TKO multiplier)
- `w_sub`: 1.0-2.0 (Submission multiplier)
- `w_udec`: 0.8-1.5 (Unanimous Decision multiplier)
- `w_mdec`: 0.7-1.3 (Majority Decision multiplier)
- `w_sdec`: 0.5-1.2 (Split Decision multiplier)

**What they control**: How much to scale K-factor based on fight outcome type

**Standard G-Elo values** (from research paper):
```
KO/TKO: 1.4
Submission: 1.3
Unanimous Decision: 1.0
Majority Decision: 0.9
Split Decision: 0.7
```

**Why optimize them?**:
- MMA may have different dynamics than other sports
- Data-driven weights may outperform literature values
- Interactions with K-factor and denominator matter

**GA typically finds**: Close to standard values, but with adjustments:
```
w_ko: 1.35-1.55 (slightly more decisive)
w_sub: 1.25-1.40 (similar or slightly more)
w_udec: 0.95-1.10 (near baseline)
w_mdec: 0.85-0.95 (slightly less decisive)
w_sdec: 0.65-0.85 (slightly less decisive)
```

---

## Understanding Fitness Functions

### Composite Fitness (Default)

Combines multiple metrics for balanced optimization:

```
Fitness = w_acc * Accuracy 
        + w_ll * (1 - normalized_log_loss)
        + w_bs * (1 - normalized_brier_score)
        + w_roi * normalized_roi
```

**Components**:

1. **Accuracy** (weight: 0.3)
   - Fraction of correct predictions
   - Range: 0.0 (all wrong) to 1.0 (all correct)
   - Simple, interpretable, but ignores confidence

2. **Log Loss / Cross-Entropy** (weight: 0.2)
   - Measures probability calibration
   - Lower is better (inverted in fitness)
   - Penalizes confident wrong predictions heavily
   - Range: 0.0 (perfect) to ∞ (terrible)

3. **Brier Score** (weight: 0.2)
   - Mean squared error of probabilities
   - Lower is better (inverted in fitness)
   - Balanced penalty for miscalibration
   - Range: 0.0 (perfect) to 1.0 (worst)

4. **ROI** (weight: 0.3)
   - Return on investment from betting
   - Based on confidence threshold
   - Range: -1.0 (lose all bets) to ~0.5 (very profitable)

**Why use composite fitness?**
- Single metrics can be misleading:
  - High accuracy but poor calibration → bad for betting
  - Good calibration but low accuracy → not useful
  - High ROI on small sample → overfitting
- Composite fitness ensures well-rounded performance
- Weights can be adjusted based on priorities

---

### ROI Fitness (Time-Series)

Optimizes specifically for betting profitability:

```
Fitness = mean(ROI_per_window) - 0.1 * std(ROI_per_window)
```

**Process**:
1. Split data into time-based windows (e.g., 6-month periods)
2. For each window:
   - Train on all data before window
   - Test on data in window
   - Calculate ROI from confident bets
3. Average ROI across windows
4. Penalize high variance (prefer stable returns)

**Advantages**:
- Directly optimizes for money-making
- Time-series validation prevents look-ahead bias
- Variance penalty prevents lucky overfitting
- Tests robustness across different time periods

**When to use**:
- Primary goal is betting profit
- Want parameters that work consistently
- Have sufficient historical data (2+ years)

---

### Accuracy-Only Fitness (Simple)

Just maximizes prediction accuracy:

```
Fitness = Accuracy
```

**Advantages**:
- Simple, easy to understand
- Fast to compute
- Good for initial exploration

**Disadvantages**:
- Ignores probability calibration
- May not optimize for betting
- Can overfit to validation set

**When to use**:
- Learning the system
- Quick experimentation
- Building pure prediction model

---

## Interpreting Results

### Convergence Plots

Generated by `full_genetic_with_k_denom_mov.py` and saved to `images/ga_convergence.png`.

**Left plot - Fitness over Generations**:
- **Best fitness** (green): Highest fitness individual each generation
- **Average fitness** (blue): Mean fitness of population
- **Worst fitness** (red): Lowest fitness individual each generation

**What to look for**:
- ✅ Best fitness increasing over time
- ✅ Average fitness converging toward best
- ✅ Population diversity (gap between best and worst) decreasing
- ⚠️ Flat best fitness for 10+ generations → converged or stuck
- ⚠️ Erratic jumps → try lower mutation rate
- ⚠️ No improvement → fitness function may be too noisy

**Right plot - Best Fitness Progression**:
- Zoomed view of best fitness only
- Shows incremental improvements
- Useful for detecting plateaus

---

### Parameter Evolution Plots

Generated by `full_genetic_with_k_denom_mov.py` and saved to `images/ga_parameter_evolution.png`.

Shows how each parameter changed over generations:

**What to look for**:
- ✅ Parameters converging to stable values → optimization successful
- ✅ Smooth progression → good exploration-exploitation balance
- ⚠️ Parameter at boundary (min/max) → may need wider search range
- ⚠️ Wild oscillations → try lower mutation rate or larger population
- ⚠️ Multiple parameters always at boundaries → bounds may be wrong

**Example interpretation**:
```
K-factor: Started ~250, converged to ~150
→ GA found lower K-factor works better

Denominator: Started ~400, converged to ~420
→ GA found slightly higher sensitivity works better

Confidence threshold: Started ~90, converged to ~75
→ GA found moderate selectivity is optimal
```

---

### ROI Results Plots

Generated by `ga_time_split_roi.py` and saved to `images/{prefix}_results.png`.

Four subplots:

1. **ROI over Generations** (top-left)
   - Best and average ROI per generation
   - Should show improvement over time

2. **Parameter Evolution** (top-right, bottom-left)
   - K-factor, denominator, confidence threshold
   - Shows convergence to optimal values

3. **ROI Distribution** (bottom-right)
   - Histogram of ROI values from different windows
   - Should be mostly positive for good strategy
   - Look for consistency (low variance)

---

### History CSV Files

All optimization runs save history to CSV files:

**Columns**:
- `generation`: Generation number (0, 1, 2, ...)
- `best_fitness`: Best fitness score in this generation
- `avg_fitness`: Average fitness across population
- `worst_fitness`: Worst fitness in this generation
- `best_genes`: JSON string with best parameter values
- Additional columns for individual parameters

**Usage**:
```python
import pandas as pd
history = pd.read_csv('ga_optimization_history.csv')

# Plot custom metric
import matplotlib.pyplot as plt
plt.plot(history['generation'], history['best_fitness'])
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()

# Extract final parameters
import json
final_params = json.loads(history.iloc[-1]['best_genes'])
print(f"Final K: {final_params['k']}")
```

---

### Best Parameters File

ROI optimization saves a text file with best parameters:

```
BEST PARAMETERS FOR ROI OPTIMIZATION
==================================================

k: 134.5623
denominator: 412.3456
confidence_threshold: 89.2341

Best ROI: 0.1234
```

**How to use**:
1. Copy parameters to your Elo calculation code
2. Run backtest to verify performance
3. Use for live predictions with same parameters
4. Monitor performance and re-optimize periodically

---

## Advanced Configuration

### Customizing Parameter Bounds

Edit the script to change search ranges:

```python
# Narrower search for K-factor if you know approximate range
param_bounds = {
    'k': (50, 200),           # Focus search on medium K values
    'denominator': (350, 450), # Focus on near-standard sensitivity
}

# Wider search if you want more exploration
param_bounds = {
    'k': (5, 1000),           # Very wide range
    'denominator': (100, 800), # Very wide range
}
```

**Tips**:
- Start wide, then narrow based on results
- If optimal is at boundary, expand that direction
- Domain knowledge can help set reasonable bounds

---

### Adjusting GA Hyperparameters

**Population Size** (`population_size`):
- Default: 50 (full optimization), 20 (examples)
- Larger = more exploration, slower per generation
- Smaller = faster, but may miss optimal solutions
- Recommended: 30-100 depending on parameter count

**Generations** (`generations`):
- Default: 100 (full optimization), 20 (examples)
- More generations = more optimization time
- Use with early stopping to avoid wasted computation
- Recommended: 50-200 depending on complexity

**Elite Size** (`elite_size`):
- Default: 5 (full optimization), 3 (examples)
- Number of best individuals preserved each generation
- Too small = may lose good solutions
- Too large = reduces exploration
- Recommended: 5-10% of population size

**Mutation Rate** (`mutation_rate`):
- Default: 0.15 (15%)
- Probability of mutating each gene
- Higher = more exploration, slower convergence
- Lower = faster convergence, may get stuck
- Recommended: 0.10-0.20 for most problems

**Crossover Rate** (`crossover_rate`):
- Default: 0.8 (80%)
- Probability of combining parent parameters
- Higher = more recombination of solutions
- Lower = more reliance on mutation
- Recommended: 0.7-0.9 for most problems

**Early Stop Generations** (`early_stop_generations`):
- Default: 15 (full optimization), 10 (examples)
- Stop if no improvement for this many generations
- Saves computation time when converged
- Recommended: 10-25 generations

---

### Custom Fitness Weights

For `full_genetic_with_k_denom_mov.py`, edit the fitness weights:

```python
# Accuracy-focused (research, prediction quality)
fitness_weights = {
    'accuracy': 0.4,
    'log_loss': 0.3,
    'brier_score': 0.3,
    'roi': 0.0
}

# Profit-focused (betting, money-making)
fitness_weights = {
    'accuracy': 0.15,
    'log_loss': 0.2,
    'brier_score': 0.15,
    'roi': 0.5
}

# Calibration-focused (probability quality)
fitness_weights = {
    'accuracy': 0.2,
    'log_loss': 0.4,
    'brier_score': 0.4,
    'roi': 0.0
}

# Balanced (default)
fitness_weights = {
    'accuracy': 0.3,
    'log_loss': 0.2,
    'brier_score': 0.2,
    'roi': 0.3
}
```

**Constraint**: Weights should sum to ~1.0 (not enforced, but recommended)

---

### Time-Series Split Configuration

For `ga_time_split_roi.py`, adjust time-based validation:

**Split Months** (`--split-months`):
```bash
# Shorter windows - faster meta changes, recent data emphasis
python optimization/ga_time_split_roi.py --split-months 3

# Standard - balanced between recency and stability
python optimization/ga_time_split_roi.py --split-months 6

# Longer windows - more stable estimates, less sensitive to trends
python optimization/ga_time_split_roi.py --split-months 12
```

**Trade-offs**:
- **Short splits (3 months)**:
  - Pro: Adapts to recent changes in MMA meta
  - Pro: More validation windows (more robust)
  - Con: Each window has less data (noisier estimates)
  
- **Long splits (12 months)**:
  - Pro: More data per window (stable estimates)
  - Pro: Better for long-term betting strategies
  - Con: Fewer validation windows (less robust)
  - Con: May not capture recent meta changes

**Recommendation**: Start with 6 months, adjust based on:
- Available data volume (need 50+ fights per window)
- How quickly MMA meta changes
- Your betting time horizon

---

### Running on Custom Data

All scripts can be adapted for custom fight data:

**Requirements**:
- CSV file with columns:
  - `FIGHTER`: Fighter 1 name
  - `opp_FIGHTER`: Fighter 2 name (opponent)
  - `result`: Outcome (1 = Fighter 1 wins, 0 = Fighter 2 wins)
  - `DATE`: Fight date (YYYY-MM-DD format)
  - Optional: `METHOD` for MOV (e.g., "KO", "Submission", "Decision")

**Adaptation**:
```bash
# For ga_time_split_roi.py
python optimization/ga_time_split_roi.py --data-file path/to/your/data.csv

# For other scripts, edit the file path in __main__ section:
# df = pd.read_csv("path/to/your/data.csv", low_memory=False)
```

---

## Tips and Best Practices

### 1. Start Simple, Then Expand

```bash
# Step 1: Learn with basic example (5 min)
python examples/example_ga_optimization.py

# Step 2: Try ROI optimization (30 min)
python optimization/ga_time_split_roi.py

# Step 3: Full multi-parameter optimization (60 min)
python optimization/full_genetic_with_k_denom_mov.py
```

### 2. Verify Results with Out-of-Sample Testing

Always test optimized parameters on completely separate data:

```python
# Don't use same data for optimization AND final evaluation
# Split data: 80% optimization, 20% held-out test set

train_df = df[df['DATE'] < split_date]
test_df = df[df['DATE'] >= split_date]

# Optimize on train_df only
# Test final parameters on test_df
```

### 3. Re-optimize Periodically

MMA meta changes over time (rule changes, training evolution):
- Re-optimize every 6-12 months
- Compare new parameters with old
- If significantly different, MMA meta has shifted

### 4. Monitor for Overfitting

**Red flags**:
- Training fitness >> validation fitness (>5% gap)
- ROI on training data >> ROI on held-out data
- Parameters at extreme boundaries
- Extremely high mutation rate needed to improve

**Solutions**:
- Use time-series cross-validation (ROI script does this)
- Increase validation data percentage
- Add regularization to fitness function
- Simplify parameter space (fix some parameters)

### 5. Compare Against Baselines

Always compare GA results with simpler approaches:
- Default Elo (K=32, denom=400)
- Grid search on K only
- Literature values for MOV weights

GA should show clear improvement to justify complexity.

### 6. Save Your Results

Keep a log of optimization runs:
```bash
# Create results directory
mkdir -p optimization_results/$(date +%Y%m%d)

# Run with date-stamped output
python optimization/ga_time_split_roi.py \
  --output-prefix optimization_results/$(date +%Y%m%d)/roi_opt
```

### 7. Parallel Execution (Advanced)

For very large datasets, consider parallel evaluation:

```python
# In ga_engine.py, modify evaluate_population method to use multiprocessing
from multiprocessing import Pool

def evaluate_population_parallel(self, population, n_processes=4):
    with Pool(n_processes) as pool:
        fitnesses = pool.map(self.fitness_fn, [ind.genes for ind in population])
    for ind, fitness in zip(population, fitnesses):
        ind.fitness = fitness
```

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'pandas'"

**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

---

### Problem: "FileNotFoundError: data/interleaved_cleaned.csv"

**Solution**: Ensure you're in the repository root and data file exists
```bash
cd /path/to/elo_research
ls data/  # Should show interleaved_cleaned.csv
```

---

### Problem: Optimization is very slow

**Solutions**:
1. Reduce dataset size for testing:
   ```python
   df = df.tail(5000)  # Use last 5000 fights only
   ```

2. Reduce population size and generations:
   ```python
   population_size=20,
   generations=30,
   ```

3. Reduce parameter space:
   ```python
   # Optimize fewer parameters
   param_bounds = {
       'k': (50, 200),
       'denominator': (350, 450),
       # Skip MOV weights initially
   }
   ```

---

### Problem: Fitness not improving after initial generations

**Possible causes and solutions**:

1. **Converged to local optimum**:
   - Increase mutation rate: `mutation_rate=0.2`
   - Increase population size: `population_size=100`
   - Widen parameter bounds

2. **Fitness function too noisy**:
   - Increase validation data size
   - Use time-series averaging (ROI script does this)
   - Add smoothing to fitness calculation

3. **Parameters at boundary**:
   - Check if best parameters are at min/max bounds
   - Expand bounds in that direction
   - May indicate unbounded optimum

---

### Problem: Results don't reproduce

**Causes**:
- Random initialization
- Stochastic selection/crossover/mutation

**Solutions**:
```python
# Set random seed for reproducibility
ga = GeneticAlgorithm(
    ...,
    random_seed=42,  # Any fixed integer
    ...
)
```

---

### Problem: High variance in results across runs

**Solution**: Run multiple optimizations and average:
```bash
# Run 5 independent optimizations
for i in {1..5}; do
  python optimization/ga_time_split_roi.py --output-prefix run_${i}
done

# Analyze consistency across runs
# Parameters should be similar if optimization is stable
```

---

## References

### Research Papers

1. **G-Elo: Generalization of the Elo Algorithm** (Szczecinski & Djebbi)
   - Introduces Method of Victory (MOV) concept
   - Provides theoretical foundation for MOV weights

2. **Genetic Algorithms in Search, Optimization, and Machine Learning** (Goldberg)
   - Classic GA textbook
   - Explains selection, crossover, mutation operators

### Code Documentation

- `optimization/ga_engine.py` - Core GA implementation
- `optimization/full_genetic_with_k_denom_mov.py` - Multi-parameter optimization
- `optimization/ga_time_split_roi.py` - ROI-focused optimization
- `elo/calculator.py` - Elo calculation logic
- `elo/elo_utils.py` - Utility functions including MOV scaling

### Further Reading

- Main README.md - Project overview and basic usage
- examples/ directory - Working code examples
- tests/ directory - Unit tests showing expected behavior

---

## Summary

This guide covered:

✅ **What** genetic algorithms are and why they're better than grid search

✅ **Three main scripts** for different optimization goals

✅ **Complete CLI commands** with all arguments explained

✅ **Usage examples** from beginner to advanced

✅ **Parameter explanations** - what each parameter does and typical ranges

✅ **Fitness functions** - different optimization objectives

✅ **Result interpretation** - understanding plots and output files

✅ **Advanced configuration** - customizing for your specific needs

✅ **Troubleshooting** - solutions to common problems

### Quick Start Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run basic example: `python examples/example_ga_optimization.py`
- [ ] Run ROI optimization: `python optimization/ga_time_split_roi.py`
- [ ] Run full optimization: `python optimization/full_genetic_with_k_denom_mov.py`
- [ ] Review plots in `images/` directory
- [ ] Check history CSV files
- [ ] Apply optimized parameters to your Elo calculations
- [ ] Validate on held-out test data

---

**Questions or issues?** Check the troubleshooting section or open an issue on GitHub.

**Want to contribute?** Feel free to submit improvements to the GA optimization system!
