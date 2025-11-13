# Ground Up Elo

An Elo rating system for fighter predictions, using genetic algorithm optimization to find the optimal K-factor.

## Overview

This project implements an Elo rating system to predict fight outcomes. It includes:

- Basic Elo rating calculation with configurable K-factor
- Genetic algorithm to optimize the K-factor based on prediction accuracy
- Out-of-sample testing on future events
- Visualization of accuracy trends over time

## Files

- `main.py` - Main Elo implementation with visualization functions
- `genetic_algorithm_k.py` - Genetic algorithm for K-factor optimization and out-of-sample testing
- `interleaved_cleaned.csv` - Historical fight data for training
- `past3_events.csv` - Recent events for out-of-sample testing

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
python main.py
```

Run the genetic algorithm optimization:
```bash
python genetic_algorithm_k.py
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
1.00, & \text{Decision} \\
1.10, & \text{TKO} \\
1.30, & \text{Submission} \\
0.90, & \text{Majority Decision} \\
0.60, & \text{Split Decision}
\end{cases}
$$

So the new update would be:

$R_1' = R_1 + K_{\text{eff}}*(S_1 - E_1)$

$R_2' = R_2 + K_{\text{eff}}*(S_2 - E_2)$


#### Prediction

We predict Fighter 1 wins if $R_1 > R_2$, and Fighter 2 wins otherwise. Predictions are only made when both fighters have at least one prior fight in the historical data.

### K-Factor Optimization

Despite the name "genetic_algorithm_k", the code actually performs a **grid search** over K values to find the optimal K-factor.

#### Training and Validation Split

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
   - These frozen ratings are used to predict outcomes on completely separate events (e.g., `past3_events.csv`)
   - This provides a true measure of generalization to future fights

This approach helps prevent overfitting: by optimizing K based on future accuracy within the historical data, we select a K-factor that generalizes well to truly unseen events.

## Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy

