# Ground Up Elo

An Elo rating system for fighter predictions, using genetic algorithm optimization to find the optimal K-factor.

## Overview

This project implements an Elo rating system to predict fight outcomes. It includes:

- Basic Elo rating calculation with configurable K-factor
- Genetic algorithm to optimize the K-factor based on prediction accuracy or ROI
- Method of Victory (MOV) weighting for more informative rating updates
- Out-of-sample testing on future events
- Visualization of accuracy trends over time
- Ensemble machine learning models for advanced predictions
- Comprehensive prediction calibration and consistency metrics
- Diagnostic analysis tools to identify prediction weak spots
- ROI-based optimization and betting quality metrics

## Files

### Core Scripts

- `scripts/main.py` - Main Elo implementation with visualization functions and ROI calculation
- `scripts/elo_utils.py` - Utility functions for Elo calculations and Method of Victory scaling
- `scripts/full_genetic_with_k_denom_mov.py` - Genetic algorithm for optimizing K-factor and Method of Victory weights

### Advanced Analysis Scripts

- `scripts/predict_fights.py` - Advanced fight prediction system combining Elo ratings with statistical features using ensemble machine learning models (Logistic Regression, Random Forest, Gradient Boosting)
- `scripts/prediction_metrics.py` - Comprehensive prediction calibration and consistency metrics including ECE (Expected Calibration Error), Brier Score, Log Loss, AUC-ROC, and betting quality metrics (Kelly Criterion, Sharpe Ratio, max drawdown)
- `scripts/optimal_k_with_mov.py` - Grid search for K-factor optimization with Method of Victory (MOV) comparison and visualization
- `scripts/analyze_baseline_diagnostics.py` - Diagnostic analysis for baseline Elo predictions across multiple dimensions (experience level, layoff duration, fight method, opponent quality, Elo margin, time trends, calibration)
- `scripts/diagnostic_tests.py` - Diagnostic test suite for ROI performance degradation analysis (rolling OOS backtest, parameter robustness, bootstrap analysis, market shift detection, Monte Carlo simulation)
- `scripts/run_all_config_tests.py` - Automated testing script to run the genetic algorithm with all 8 combinations of feature flags (multiphase-decay, weight-adjust, optimize-elo-denom) and generate comparison reports

### Test Files

- `scripts/test_ga_roi.py` - Unit tests for ROI-based genetic algorithm functions
- `scripts/test_prediction_metrics.py` - Unit tests for prediction calibration and consistency metrics
- `scripts/test_roi_calculator.py` - Unit tests for ROI calculator functions

### Data Files

- `data/interleaved_cleaned.csv` - Historical fight data for training
- `data/past3_events.csv` - Recent events for out-of-sample testing
- `data/most_recent_elo.csv` - Latest Elo ratings for all fighters

### Generated Images

- `images/mov_comparison_plot.png` - MOV vs No MOV accuracy comparison across K values
- `images/baseline_diagnostics.png` - Diagnostic analysis results visualization
- `images/k_optimization_accuracy_plot.png` - K-factor optimization accuracy plot
- `images/roi_over_time.png` - ROI performance over time

## Setup

1. Create a virtual environment:
```bash
python3 -m venv elo_env
source elo_env/bin/activate  # On Windows: elo_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main Elo analysis:
```bash
python scripts/main.py
```

Run the genetic algorithm optimization (optimizes both K-factor and MOV weights):
```bash
python scripts/full_genetic_with_k_denom_mov.py
```

Run the MOV comparison analysis (compares Elo with and without Method of Victory weights using grid search):
```bash
python scripts/optimal_k_with_mov.py
```

### Advanced Analysis

Run the advanced fight prediction system using ensemble models:
```bash
python scripts/predict_fights.py
```

Run diagnostic analysis on baseline Elo predictions to identify weak spots:
```bash
python scripts/analyze_baseline_diagnostics.py
python scripts/analyze_baseline_diagnostics.py --output-json results/diagnostics.json
python scripts/analyze_baseline_diagnostics.py --no-visualizations
```

Run diagnostic tests for ROI performance degradation analysis:
```bash
python scripts/diagnostic_tests.py           # Run all tests
python scripts/diagnostic_tests.py --test 1  # Run specific test (1-6)
python scripts/diagnostic_tests.py --all     # Run all tests with verbose output
```

Run all configuration tests (tests all 8 feature flag combinations):
```bash
python scripts/run_all_config_tests.py
python scripts/run_all_config_tests.py --seed 42
python scripts/run_all_config_tests.py --generations 5 --population 10
```

### Running Tests

Run all unit tests:
```bash
python -m pytest scripts/test_*.py -v
```

Or run individual test files:
```bash
python scripts/test_ga_roi.py
python scripts/test_prediction_metrics.py
python scripts/test_roi_calculator.py
```

### Updating GitHub Repository

To automatically add, commit, and push all changes to GitHub:

```bash
./push_to_github.sh
```

Or with a custom commit message:
```bash
./push_to_github.sh "Your commit message here"
```

The script will:
1. Show current changes
2. Stage all files
3. Commit with your message (or a timestamped default)
4. Push to GitHub

## Methods/Math

### Elo Rating System

The Elo rating system assigns each fighter a numerical rating that reflects their skill level. After each fight, ratings are updated based on the outcome versus the expected outcome.

#### Expected Score

Before a fight, we calculate the expected probability that Fighter 1 wins:

### $E_1 = \frac{1}{1 + 10^{(R_2 - R_1) / 400}}$


where:
- $R_1$ is Fighter 1's current Elo rating
- $R_2$ is Fighter 2's current Elo rating
- The denominator (400) controls the sensitivity: a 400-point difference means one fighter is 10x more likely to win

#### Rating Updates

After the fight, ratings are updated:


### $R_1^{new} = R_1^{old} + K \cdot (S_1 - E_1)$


### $R_2^{new} = R_2^{old} + K \cdot (S_2 - E_2)$

where:
- $S_1$ is the actual outcome (1 for Fighter 1 win, 0 for loss)
- $S_2 = 1 - S_1$
- $K$ is the **K-factor** that determines how much ratings change after each fight

A higher K-factor means ratings update more quickly but can be more volatile. A lower K-factor means more stable ratings but slower adaptation to changes in fighter ability.

#### Method of Victory Optimization

G-Elo: Generalized Elo using margin of victory (Szczecinski), is a research paper that proposes a slight change to the Elo algorithm by incorporating margin of victory (MOV) rather than only win/loss. In the MMA case, we would want to include things like unanimous decision vs split decision vs submission. 

##### Math change:

Right now the update is: 

### $$R_1' = R_1 + K*(S_1 - E_1)$$
### $$R_2' = R_2 + K*(S_2 - E_2)$$

Now we change K to be 

### $K_{\text{eff}} = K * M(fight)$ 


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

### $R_1' = R_1 + K_{\text{eff}}*(S_1 - E_1)$

### $R_2' = R_2 + K_{\text{eff}}*(S_2 - E_2)$


#### Prediction

We predict Fighter 1 wins if $R_1 > R_2$, and Fighter 2 wins otherwise. Predictions are only made when both fighters have at least one prior fight in the historical data.

### K-Factor Optimization

This project provides two approaches for optimizing Elo parameters:

#### Grid Search (optimal_k_with_mov.py)

Despite the function name "genetic_algorithm_k", this script performs a **grid search** over K values to find the optimal K-factor.

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

#### Genetic Algorithm (full_genetic_with_k_denom_mov.py)

The genetic algorithm simultaneously optimizes both the K-factor and all Method of Victory (MOV) weights, providing a more sophisticated search through the parameter space.

**Optimized Parameters:**
- K-factor (range: 10.0 to 500.0)
- KO/TKO weight (range: 1.0 to 2.0)
- Submission weight (range: 1.0 to 2.0)
- Unanimous Decision weight (range: 0.8 to 1.2)
- Split Decision weight (range: 0.5 to 1.1)
- Majority Decision weight (range: 0.7 to 1.2)

**Algorithm Components:**

1. **Population Initialization**: Creates a population of random parameter sets (default: 30 individuals)

2. **Fitness Evaluation**: For each parameter set:
   - Runs Elo with the given K and MOV weights
   - Evaluates prediction accuracy on the validation set (last 20% of training data by date)
   - Uses validation accuracy as the fitness score

3. **Selection**: Uses tournament selection (default: tournament size of 3)
   - Randomly selects k individuals from the population
   - Returns the individual with the highest fitness

4. **Crossover**: Creates offspring by combining parent parameters
   - For each parameter, with probability `crossover_rate` (default: 0.5), takes the average of both parents
   - Otherwise, randomly selects a value from one parent

5. **Mutation**: Applies Gaussian noise to parameters
   - Each parameter has a `mutation_rate` (default: 0.3) chance of being mutated
   - Mutation adds noise proportional to the parameter's range scaled by `mutation_scale` (default: 0.1)
   - Mutated values are clipped to stay within parameter bounds

6. **Elitism**: Preserves the best individual from each generation to the next

7. **Evolution Loop**: Repeats for a specified number of generations (default: 30)
   - Each generation: select parents, create offspring via crossover and mutation, evaluate fitness
   - Tracks the best parameters found across all generations

**Advantages of Genetic Algorithm:**
- Searches a high-dimensional parameter space (6 parameters) more efficiently than grid search
- Can discover non-obvious parameter combinations that work well together
- Balances exploration (mutation) and exploitation (selection of good solutions)
- Particularly useful when optimizing multiple interdependent parameters like MOV weights

**Output:**
The algorithm returns the best parameter set found, along with its validation accuracy. These parameters can then be used for out-of-sample testing on truly unseen events.

## Results

### Method of Victory (MOV) Impact

We compared the Elo rating system with and without Method of Victory weights to evaluate the impact of incorporating fight outcome decisiveness into the rating updates.

#### Summary Comparison

**WITH MOV:**
- Best K: 170
- Best Test Accuracy: 0.5861
- OOS Accuracy (at best K): 0.6053

**WITHOUT MOV:**
- Best K: 250
- Best Test Accuracy: 0.5789
- OOS Accuracy (at best K): 0.5789

**MOV Improvement:**
- Test Accuracy: +0.0072 (1.2% improvement)
- OOS Accuracy: +0.0264 (4.6% improvement)

The results show that incorporating Method of Victory weights provides meaningful improvements, particularly in out-of-sample accuracy, demonstrating better generalization to future fights.

#### K Parameter Optimization and MOV Comparison

The following plot compares the Elo rating system with and without Method of Victory (MOV) weights across different K values:

![MOV Comparison Plot](images/mov_comparison_plot.png)

**Plot Breakdown:**

The visualization consists of four subplots comparing MOV vs No MOV across different accuracy metrics:

1. **Top-Left: Overall Accuracy**
   - Both "With MOV" (blue circles) and "Without MOV" (orange squares) lines are nearly identical
   - Both consistently hover around 0.56-0.57 accuracy across all K values
   - **Finding**: MOV has negligible impact on overall accuracy

2. **Top-Right: Test Accuracy (Future)**
   - "With MOV" (blue circles) peaks around 0.58-0.59 for K values 150-200
   - "Without MOV" (orange squares) peaks around 0.58 for K values 200-250
   - "With MOV" maintains slightly better performance in the optimal K range
   - **Finding**: MOV provides a modest improvement in test accuracy, with MOV preferring lower K values

3. **Bottom-Left: Out-of-Sample Accuracy**
   - "With MOV" (blue triangles) shows a strong peak of ~0.63-0.64 for K values 180-280
   - "Without MOV" (orange inverted triangles) drops sharply to ~0.52-0.53 for K values 180-200, then recovers to ~0.58-0.59
   - **Finding**: MOV demonstrates a clear advantage in out-of-sample accuracy, achieving substantially higher peak performance where No MOV performs poorly

4. **Bottom-Right: All Metrics Combined**
   - Provides a consolidated view of all six accuracy metrics
   - Clearly shows that MOV's primary benefit is in out-of-sample accuracy
   - The OOS accuracy divergence is the most pronounced difference between the two approaches

**Key Takeaways:**
- MOV has minimal impact on overall accuracy but provides meaningful improvements in test and out-of-sample accuracy
- The optimal K value differs: MOV performs best at K=170, while No MOV performs best at K=250
- MOV is particularly effective for predicting truly unseen events (out-of-sample), achieving up to 63% accuracy compared to No MOV's peak of ~60%
- The improvement is most pronounced in the K range of 180-280, where MOV maintains high OOS accuracy while No MOV experiences a performance dip

### Genetic Algorithm Optimization Results

The genetic algorithm simultaneously optimized both the K-factor and all Method of Victory weights, searching a 6-dimensional parameter space to find optimal combinations.

#### Best Parameters Found

After 30 generations with a population size of 30:

- **K-factor**: 266.33
- **KO/TKO weight**: 1.02
- **Submission weight**: 1.74
- **Unanimous Decision weight**: 0.98
- **Split Decision weight**: 0.70
- **Majority Decision weight**: 0.81

#### Performance Metrics

- **Validation Accuracy** (last 20% of training data): 59.60%
- **Out-of-Sample Accuracy** (truly unseen events from 2025): 57.14%
- **Gap Accuracy** (fights with Elo difference ≥ 75): 53.85%

#### Key Insights

1. **Optimal K-factor**: The GA found K ≈ 266, which is higher than the grid search optimal of K=170. This suggests that when MOV weights are optimized simultaneously, a higher K-factor works better.

2. **MOV Weight Patterns**:
   - **Submissions** receive the highest weight (1.74), indicating they are the most decisive outcomes
   - **KO/TKO** weight is near baseline (1.02), suggesting they're already well-captured
   - **Split decisions** receive the lowest weight (0.70), reflecting their less decisive nature
   - **Unanimous decisions** are close to baseline (0.98), while **majority decisions** are slightly below (0.81)

3. **Generalization**: The OOS accuracy of 57.14% is above random chance (50%) and close to the validation accuracy (59.60%), indicating good generalization. The smaller OOS sample size (14 fights) reflects the temporal gap between training data (ending 2022-07-27) and test events (2025), where many fighters lack prior history in the training period.

4. **Convergence**: The GA converged by generation 29, finding a stable solution that balances exploration and exploitation of the parameter space.

## Advanced Features

### Ensemble Fight Prediction (predict_fights.py)

The advanced fight prediction system combines Elo ratings with statistical features using multiple machine learning models for improved accuracy.

**Features Extracted:**
- **Elo Features**: Elo difference, average Elo, Elo ratio between fighters
- **Physical Attributes**: Height difference, reach difference, age difference
- **Experience Features**: Bout count difference, historical win rate difference
- **Striking Features**: Significant strikes per minute, striking accuracy, striking defense
- **Grappling Features**: Takedown average, takedown accuracy, takedown defense
- **Recent Form**: Win rate and striking performance over last 3 fights

**Ensemble Models:**
- Logistic Regression (with feature scaling)
- Random Forest Classifier (100 estimators, max depth 10)
- Gradient Boosting Classifier (100 estimators, max depth 5)

The ensemble uses voting (average probabilities) to make final predictions.

### Prediction Calibration & Consistency Metrics (prediction_metrics.py)

Comprehensive metrics for evaluating prediction quality beyond simple accuracy:

**Calibration Metrics:**
- **ECE (Expected Calibration Error)**: Measures how well predicted probabilities match actual win rates
- **Brier Score**: Mean squared error between predictions and outcomes (0 = perfect, 1 = worst)
- **Log Loss**: Cross-entropy loss measuring prediction confidence
- **Calibration Slope**: Linear regression of actual vs predicted (slope = 1.0 is perfect)

**Consistency Metrics:**
- **By Elo Tier**: Accuracy variance across different Elo difference buckets
- **By Fight Method**: Performance across KO/TKO, Submission, Decision outcomes
- **By Time Period**: Monthly accuracy trends to detect drift
- **By Experience Gap**: Performance when fighters have different experience levels

**Betting Quality Metrics:**
- **Kelly Criterion**: Optimal bet sizing based on edge
- **Max Drawdown**: Largest peak-to-trough decline in cumulative profits
- **Sharpe Ratio**: Risk-adjusted return by time period
- **Value Bet Analysis**: Identifies bets where model probability exceeds implied odds

### Diagnostic Analysis (analyze_baseline_diagnostics.py)

Comprehensive diagnostic analysis to identify which prediction subgroups have poor accuracy/ROI:

**Analysis Dimensions:**
1. **By Fighter Experience**: Unknown, Novice, Experienced, Veteran
2. **By Layoff Duration**: Recent (0-90 days), Medium (91-180 days), Long (181-274 days), Very Long (275+ days)
3. **By Fight Method**: KO/TKO, Submission, Decision types
4. **By Opponent Quality**: Lower-tier (<1500 Elo), Mid-tier (1500-1700), Elite (>1700)
5. **By Elo Margin**: Huge favorite, Large favorite, Close, Underdog categories
6. **Over Time**: Monthly trends to detect model drift
7. **Confidence Calibration**: ECE by prediction confidence decile

**Output:**
- Weak spots and strong spots identification
- Optimization targets ranked by potential ROI impact
- Visualization saved to `images/baseline_diagnostics.png`

### Diagnostic Test Suite (diagnostic_tests.py)

Six diagnostic tests to analyze ROI performance degradation between training and OOS:

1. **Rolling OOS Backtest**: Train on past windows, test on next window to measure typical ROI drop
2. **Parameter Robustness**: Compare best GA params vs random params on OOS data (overfitting detection)
3. **Reshuffle OOS Test**: Bootstrap ROI from random historical samples
4. **Market Shift Analysis**: Compare odds/rating distributions between training and OOS periods
5. **Bootstrap/Monte Carlo**: Build ROI variance distribution to understand expected variance
6. **Accuracy vs ROI Analysis**: Compare prediction accuracy and ROI to identify if accuracy or odds drive performance

### Configuration Testing (run_all_config_tests.py)

Automated testing across all 8 combinations of feature flags:
- **multiphase-decay**: Enhanced rating decay for activity gaps
- **weight-adjust**: Weight class adjustment feature
- **optimize-elo-denom**: Elo denominator optimization (adjusts the 400 in expected score formula)

**Output:**
- Comparison report with ROI, accuracy, and best parameters for each configuration
- JSON files with full parameter precision for reproducibility
- Copy-paste ready Python dictionaries for use in main.py

## Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy
- scikit-learn (for predict_fights.py and prediction_metrics.py)
