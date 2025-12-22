# Elo Research

An Elo rating system for MMA fighter predictions with genetic algorithm optimization.

## Structure

```
elo_research/
├── main.py              # Main entry point for Elo rankings and visualizations
├── elo/                 # Core Elo rating system modules
│   ├── calculator.py    # Elo calculation functions
│   ├── elo_utils.py     # Utility functions for Elo calculations
│   ├── time_splitter.py # Time-based data splitting
│   └── visualization.py # Visualization functions
├── optimization/        # Genetic algorithm optimization scripts
│   ├── ga_engine.py         # Core GA implementation
│   ├── full_genetic_with_k_denom_mov.py  # Multi-parameter GA optimization
│   ├── ga_time_split_roi.py # ROI-focused GA with time-series validation
│   └── optimal_k_with_mov.py # Grid search baseline (legacy)
├── analysis/            # Analysis and diagnostic scripts
│   ├── analyze_baseline_diagnostics.py
│   ├── diagnostic_tests.py
│   └── prediction_metrics.py
├── tests/               # Test scripts
├── data/                # Fight data
└── images/              # Output images
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/evankellener/elo_research.git
cd elo_research
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

This will install:
- pandas (data manipulation)
- numpy (numerical computing)
- matplotlib (visualization)
- scikit-learn (machine learning metrics)

### Data Requirements

Ensure you have the required data file in the `data/` directory:
- `data/interleaved_cleaned.csv` - Historical MMA fight data

## Quick Start

> **Note:** All commands below should be run from the repository root directory (`elo_research/`)

### 1. Basic Elo Analysis

Run the main Elo analysis and visualizations:

```bash
python main.py
```

This will:
- Calculate basic Elo ratings
- Calculate Elo with Method of Victory (MOV) weights
- Display top fighters by Elo
- Show current top rankings
- Optionally graph fighter history

#### Testing Optimized Parameters

After running GA optimization, you can test the optimized parameters in main.py:

```bash
# Test with custom K and denominator
python main.py --k 10.0 --denominator 437.78

# Calculate ROI with optimized parameters using full fighter history (RECOMMENDED)
python main.py --k 10.0 --denominator 437.78 --show-roi --use-ga-setup --tail-fights 3000

# Match exact GA optimization behavior (calculate Elo only on tail fights)
python main.py --k 10.0 --denominator 437.78 --show-roi --use-ga-setup --tail-fights 3000 --tail-only

# Adjust betting confidence threshold
python main.py --k 10.0 --denominator 437.78 --show-roi --confidence-threshold 30
```

**Important:** By default, `--tail-fights` now uses full fighter history for Elo calculations:
- **Without `--tail-only`** (NEW DEFAULT): Elo ratings are calculated from ALL historical fights, even when evaluating only recent fights. This provides more accurate ratings that reflect each fighter's complete career.
- **With `--tail-only`**: Elo ratings are calculated ONLY from the tail fights, matching the original GA optimization behavior. Use this for exact reproducibility of GA results.

The `--use-ga-setup` flag ensures the ROI calculation matches the GA optimization by:
- Adding bout count tracking
- Filtering out fights where either fighter has no prior history
- Using the same validation split (last 20% of data)

**Example - Reproducing GA Results:** If GA optimization finds K=10.0, denominator=437.78 with 34.92% ROI, running:
```bash
python main.py --k 10.0 --denominator 437.7802 --show-roi --use-ga-setup --tail-fights 3000 --tail-only
```
Will produce the same 34.92% ROI, confirming the optimization results.

**Example - Realistic Evaluation:** To see how those parameters perform with full fighter history:
```bash
python main.py --k 10.0 --denominator 437.7802 --show-roi --use-ga-setup --tail-fights 3000
```
This provides a more realistic assessment by using complete fighter career data for Elo calculations.

### 2. Running the Genetic Algorithm (GA) Optimization

The GA optimization helps you find the best parameters for your Elo rating system. Here are three ways to run it, from simplest to most advanced:

#### Option A: Quick Example (Best for Learning) ⭐ **START HERE**

Run a quick 5-minute example to understand how GA works:

```bash
python examples/example_ga_optimization.py
```

**What it does:**
- Uses 3,000 recent fights for fast execution
- Optimizes K-factor and denominator
- Shows fitness improvement over generations
- Takes ~5 minutes to complete

**Example output (actual values will vary):**
```
======================================================================
GENETIC ALGORITHM OPTIMIZATION EXAMPLE
======================================================================
Loading data...
Using 3000 fights for optimization

Running genetic algorithm...
Generation 1/20: Best=0.5823, Avg=0.5645, Worst=0.5234
Generation 5/20: Best=0.5912, Avg=0.5801, Worst=0.5623
...
Generation 20/20: Best=0.6034, Avg=0.5956, Worst=0.5823

======================================================================
RESULTS
======================================================================
Best K-factor: 72.45
Best denominator: 436.78
Best accuracy: 0.6034
```

**Note:** The exact parameter values and fitness scores will vary between runs due to the stochastic nature of genetic algorithms.

**Try different optimization modes:**
```bash
# Optimize for prediction accuracy (default)
python examples/example_ga_optimization.py --optimize-for accuracy

# Optimize for betting profit (ROI)
python examples/example_ga_optimization.py --optimize-for roi

# Optimize for probability calibration (log loss)
python examples/example_ga_optimization.py --optimize-for log_loss

# Optimize for all metrics combined
python examples/example_ga_optimization.py --optimize-for composite
```

#### Option B: Full Multi-Parameter Optimization (Production Quality)

Optimize all Elo parameters for best overall performance:

```bash
python optimization/full_genetic_with_k_denom_mov.py
```

**What it optimizes:**
- K-factor (10-500)
- Denominator (200-600)
- Confidence threshold (30-150)
- 5 Method of Victory weights (KO, Submission, Unanimous/Majority/Split Decision)

**Fitness function:** Balanced combination of accuracy, log loss, Brier score, and ROI

**Runtime:** 30-60 minutes (uses all historical data)

**Output files:**
- `ga_optimization_history.csv` - Full optimization history
- `images/ga_convergence.png` - Fitness convergence plot
- `images/ga_parameter_evolution.png` - Parameter evolution over generations

#### Option C: ROI-Focused Optimization (For Betting Strategies)

Optimize specifically for betting profitability with time-series validation:

```bash
python optimization/ga_time_split_roi.py
```

**What it does:**
- Splits data into time-based validation windows (default: 6 months)
- Optimizes K-factor, denominator, and confidence threshold
- Maximizes average ROI across all time periods
- Penalizes high variance (prefers stable returns)

**Runtime:** 20-40 minutes

**Customize settings:**
```bash
# Use 3-month windows (adapts faster to meta changes)
python optimization/ga_time_split_roi.py --split-months 3

# Use 12-month windows (more stable estimates)
python optimization/ga_time_split_roi.py --split-months 12

# Custom data file
python optimization/ga_time_split_roi.py --data-file data/my_fights.csv

# Larger population for better search
python optimization/ga_time_split_roi.py --population-size 80 --generations 150
```

**Output files:**
- `ga_roi_history.csv` - Optimization history
- `ga_roi_best_params.txt` - Best parameters found
- `images/ga_roi_results.png` - ROI visualization plots

### 3. Detailed Documentation

For comprehensive information about GA optimization, see:

📘 **[Complete GA Optimization Guide](docs/GA_OPTIMIZATION_GUIDE.md)** - Detailed explanations, parameter tuning, and advanced usage

📋 **[Quick Reference Card](docs/GA_OPTIMIZATION_QUICK_REFERENCE.md)** - Command cheat sheet

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'" (or numpy/matplotlib/sklearn)

**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: data/interleaved_cleaned.csv"

**Solution:** Ensure you're in the repository root directory and the data file exists
```bash
cd /path/to/elo_research
ls data/  # Should show interleaved_cleaned.csv
```

If the data file is missing, you need to obtain the MMA fight data and place it in the `data/` directory.

### GA optimization is too slow

**Solution 1:** Start with the quick example (uses subset of data)
```bash
python examples/example_ga_optimization.py
```

**Solution 2:** Reduce the data size in the script by editing the file to use `df.tail(5000)` instead of all data

**Solution 3:** Reduce population size and generations:
```bash
python optimization/ga_time_split_roi.py --population-size 20 --generations 30
```

### "command not found: python"

**Solution:** Try using `python3` instead:
```bash
python3 examples/example_ga_optimization.py
```

## Elo System

The Elo rating system assigns each fighter a numerical rating that reflects their skill level. After each fight, ratings are updated based on the outcome versus the expected outcome.

### Expected Score

Before a fight, we calculate the expected probability that Fighter 1 wins:

$$E_1 = \frac{1}{1 + 10^{(R_2 - R_1) / 400}}$$

where:
- $R_1$ is Fighter 1's current Elo rating
- $R_2$ is Fighter 2's current Elo rating
- The denominator (400) controls the sensitivity: a 400-point difference means one fighter is 10x more likely to win

### Rating Updates

After the fight, ratings are updated:

$$R_1^{new} = R_1^{old} + K \cdot (S_1 - E_1)$$

$$R_2^{new} = R_2^{old} + K \cdot (S_2 - E_2)$$

where:
- $S_1$ is the actual outcome (1 for Fighter 1 win, 0 for loss)
- $S_2 = 1 - S_1$
- $K$ is the **K-factor** that determines how much ratings change after each fight

A higher K-factor means ratings update more quickly but can be more volatile. A lower K-factor means more stable ratings but slower adaptation to changes in fighter ability.

### Method of Victory

G-Elo: Generalized Elo using margin of victory (Szczecinski), is a research paper that proposes a slight change to the Elo algorithm by incorporating margin of victory (MOV) rather than only win/loss. In the MMA case, we would want to include things like unanimous decision vs split decision vs submission.

#### Math change:

Right now the update is:

$$R_1' = R_1 + K*(S_1 - E_1)$$
$$R_2' = R_2 + K*(S_2 - E_2)$$

Now we change K to be:

$$K_{\text{eff}} = K * M(fight)$$

$$
M(\text{fight}) =
\begin{cases}
1.40, & \text{KO/TKO} \\
1.30, & \text{Submission} \\
1.00, & \text{Unanimous Decision} \\
0.90, & \text{Majority Decision} \\
0.70, & \text{Split Decision}
\end{cases}
$$

So the new update would be:

$$R_1' = R_1 + K_{\text{eff}}*(S_1 - E_1)$$
$$R_2' = R_2 + K_{\text{eff}}*(S_2 - E_2)$$

MOV weights scale the K-factor based on fight outcome:
- KO/TKO: 1.4x
- Submission: 1.3x
- Unanimous Decision: 1.0x
- Majority Decision: 0.9x
- Split Decision: 0.7x

This reflects that more decisive victories should have larger rating impacts.

### Prediction

We predict Fighter 1 wins if $R_1 > R_2$, and Fighter 2 wins otherwise. Predictions are only made when both fighters have at least one prior fight in the historical data.

### Parameter Optimization

This project provides two approaches to optimize Elo parameters:

#### Genetic Algorithm Optimization (NEW)

The genetic algorithm approach (`full_genetic_with_k_denom_mov.py` and `ga_time_split_roi.py`) uses evolutionary computation to simultaneously optimize multiple parameters:

**Parameters Optimized:**
- **K-factor**: Controls rating change magnitude (10-500)
- **Denominator**: Controls rating difference sensitivity (200-600)
- **MOV weights**: Five weights for different outcomes (KO, Sub, UD, MD, SD)
- **Confidence threshold**: For ROI optimization, minimum Elo difference to bet

**Fitness Function - Composite Metrics:**

The GA optimizes using a weighted combination of multiple metrics:
- **Accuracy** (30%): Prediction accuracy on validation set
- **Log Loss** (20%): Cross-entropy loss (lower is better)
- **Brier Score** (20%): Mean squared error of probabilities (lower is better)
- **ROI** (30%): Return on investment for confident predictions

This multi-metric approach ensures the model is well-calibrated, accurate, and profitable, rather than optimizing for just one dimension.

**How Genetic Algorithms Work:**

Unlike grid search which tests predefined values, genetic algorithms use evolutionary operators:

1. **Population**: A set of candidate solutions (parameter combinations)
2. **Selection**: Better solutions are more likely to reproduce
3. **Crossover**: Combine parameters from two parents to create offspring
4. **Mutation**: Randomly adjust parameters to explore new solutions
5. **Elitism**: Preserve the best solutions across generations

This allows the GA to:
- Explore a much larger parameter space efficiently
- Find optimal combinations that grid search would miss
- Optimize multiple parameters simultaneously (7+ parameters)
- Adapt search based on fitness landscape

**Usage Examples:**

```bash
# Optimize with composite fitness (default: accuracy, log loss, Brier, ROI)
python optimization/full_genetic_with_k_denom_mov.py

# Optimize for betting ROI with time-series validation
python optimization/ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 6

# Try different fitness weight combinations
python examples/example_composite_fitness.py
```

**Customizing Fitness Weights:**

You can adjust the relative importance of each metric by changing `fitness_weights`:

```python
# Balanced (default)
fitness_weights = {'accuracy': 0.3, 'log_loss': 0.2, 'brier_score': 0.2, 'roi': 0.3}

# Accuracy-focused (emphasize prediction quality)
fitness_weights = {'accuracy': 0.4, 'log_loss': 0.3, 'brier_score': 0.3, 'roi': 0.0}

# Profit-focused (emphasize betting returns)
fitness_weights = {'accuracy': 0.15, 'log_loss': 0.2, 'brier_score': 0.15, 'roi': 0.5}
```

**GA Configuration:**
- Population size: 50 individuals
- Generations: 100 (with early stopping)
- Selection: Tournament selection (size 3)
- Crossover: Uniform crossover (80% rate)
- Mutation: Gaussian mutation (15% rate)
- Elitism: Top 5 individuals preserved

**Output:**
- Convergence plots showing fitness improvement
- Parameter evolution over generations
- Best parameter combination
- Comparison with grid search baseline

#### Grid Search Baseline (optimal_k_with_mov.py)

The legacy approach performs a **grid search** over K values to find the optimal K-factor.

**Training and Validation Split:**

1. **Data Split**: The historical data is split at the 80th percentile by date
   - First 80%: Training data (used to calculate ratings)
   - Last 20%: Validation data (used to evaluate K-factor choices)

2. **Grid Search**: The algorithm tests K values in a range (default: 10 to 490 in steps of 10)
   - For each K value, it:
     - Runs the Elo system on all historical data
     - Calculates prediction accuracy on the validation set (last 20%)
     - Selects the K that maximizes validation accuracy

3. **Out-of-Sample Testing**: After finding the best K:
   - Ratings are frozen using only the training data
   - These frozen ratings are used to predict outcomes on completely separate events (e.g., `data/past3_events.csv`)
   - This provides a true measure of generalization to future fights

This approach helps prevent overfitting: by optimizing K based on future accuracy within the historical data, we select a K-factor that generalizes well to truly unseen events.

## Results

### Comprehensive Performance Analysis: With vs Without MOV

The current Elo system uses K=170 with Method of Victory (MOV) weights. Below is a complete comparison against the baseline system without MOV (K=250).

#### Performance Metrics

| Metric | WITH MOV (K=170) | WITHOUT MOV (K=250) | Difference | Better? |
|--------|------------------|---------------------|------------|---------|
| **Validation (Last Year - 352 fights)** |
| Accuracy | 60.23% | 60.51% | -0.28% | ❌ |
| Log Loss | 0.6787 | 0.7059 | -0.0272 | ✓ |
| Brier Score | 0.2424 | 0.2518 | -0.0094 | ✓ |
| **ROI** | **-7.12%** | **-9.43%** | **+2.31%** | **✓** |
| **Out-of-Sample (76 fights, market odds)** |
| Accuracy | 53.95% | 53.95% | +0.00% | → |
| Log Loss | 0.8452 | 0.9473 | -0.1021 | ✓ |
| Brier Score | 0.2898 | 0.3127 | -0.0229 | ✓ |
| **ROI** | **-9.63%** | **-9.96%** | **+0.33%** | **✓** |

**Note:** Lower is better for Log Loss and Brier Score. Higher is better for Accuracy and ROI.

#### Key Findings

**MOV Impact on Validation Set:**
- **Improves 3 of 4 metrics** (Log Loss, Brier Score, ROI)
- **ROI improvement: +2.31%** (24.5% relative improvement)
- Slightly lower accuracy but better calibration and profitability

**MOV Impact on Out-of-Sample:**
- **Improves 3 of 4 metrics** (Log Loss, Brier Score, ROI)
- **ROI improvement: +0.33%** (3.3% relative improvement)
- Same accuracy but significantly better probability calibration
- Better calibration translates to better betting decisions

**Overall Assessment:**
Method of Victory weights enhance the system's performance, particularly in:
1. **Probability calibration** (lower log loss and Brier score)
2. **Betting profitability** (higher ROI on both validation and OOS)
3. **Generalization** (improvements consistent across different evaluation sets)

The slightly lower accuracy with MOV is offset by substantially better probability estimates, which are more important for profitable betting than raw accuracy.

### Calibration Optimization Experiment

To test whether optimizing for calibration metrics (Sharpe ratio, ECE, Brier score) can improve generalization and reduce the validation-OOS gap, we ran two genetic algorithm optimizations:

#### Experiment Setup
- **Standard Optimization**: Equal weights on accuracy, log loss, Brier score, and ROI
- **Calibration-Focused**: Emphasized calibration (15% accuracy, 25% log loss, 30% Brier, 30% ROI) + bonus for Sharpe ratio, ECE, and calibration metrics

#### Results

| Approach | K | Denom | Val ROI | OOS ROI | Val-OOS Gap |
|----------|---|-------|---------|---------|-------------|
| **Standard** | 75.8 | 446.4 | -0.53% | -4.54% | **4.01%** |
| **Calibration-Focused** | 81.8 | 485.1 | -0.46% | -4.54% | **4.09%** |

**Finding**: Calibration-focused optimization did not reduce the validation-OOS gap in this experiment (4.09% vs 4.01%). Both approaches achieved similar calibration metrics:
- Log Loss: ~0.669 (val) vs ~0.724 (OOS)
- Brier Score: ~0.239 (val) vs ~0.262 (OOS)

**Analysis**: The val-OOS gap appears to be inherent to the data distribution differences between historical and future events, rather than an optimization artifact. The similar performance suggests:
1. Both optimizations found well-calibrated solutions
2. The gap reflects genuine differences between validation and OOS fight characteristics
3. Further improvements may require different features or modeling approaches rather than just calibration tuning

**Note**: This experiment used smaller population sizes (20) and fewer generations (30) for faster results. Larger-scale optimization might yield different insights.

### System Configuration

**Current Parameters:**
- K-factor: 170
- Method of Victory (MOV): Enabled
- Denominator: 400

The system incorporates fight outcome decisiveness (KO, submission, decision types) into rating updates, which improves prediction quality and betting performance.

### Genetic Algorithm vs Grid Search

The new genetic algorithm optimization provides significant advantages over traditional grid search:

**Why Genetic Algorithms?**

Traditional grid search over K-factor tests ~50 values in a 1D space. To similarly test 7 parameters (K, denominator, 5 MOV weights), grid search would need:
- 50^7 = 781 billion evaluations (computationally infeasible)
- GA achieves better results with ~5,000 evaluations (50 population × 100 generations)

**Advantages of GA Approach:**

1. **Multi-parameter optimization**: Simultaneously optimizes K, denominator, and MOV weights
   - Grid search: Linear search through single parameters
   - GA: Explores parameter interactions and combinations

2. **Efficiency**: Finds good solutions much faster
   - Grid search: Exhaustive, predictable runtime
   - GA: Adaptive search, early stopping when converged

3. **Solution quality**: Can find better local optima
   - Grid search: Limited by grid granularity
   - GA: Continuous parameter space with mutation

4. **Flexibility**: Easy to add new parameters or constraints
   - **ROI optimization**, decay rates, weight class adjustments
   - Time-series cross-validation for robustness

The GA explores unconventional parameter combinations (like lower K with higher denominator) that grid search would never test, leading to better betting performance.

## Requirements

See `requirements.txt` for dependencies.
