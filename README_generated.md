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

`E_1 = 1 / (1 + 10^((R_2 - R_1) / 400))`

Where:
- R1 is Fighter 1's current Elo rating
- R2 is Fighter 2's current Elo rating
- 400 controls sensitivity (400-point difference ≈ 10x win likelihood)

#### Rating Updates

After the fight:

`R_1_new = R_1_old + K * (S_1 - E_1)`
`R_2_new = R_2_old + K * (S_2 - E_2)`

Where:
- S1 = 1 for Fighter 1 win, 0 for loss
- S2 = 1 - S1
- K controls rating volatility

#### Method of Victory Optimization

G-Elo modifies Elo to include a margin-of-victory multiplier.

Original update:

`R_1' = R_1 + K * (S_1 - E_1)`
`R_2' = R_2 + K * (S_2 - E_2)`

MOV multiplier table:

Decision: 1.00  
TKO: 1.10  
Submission: 1.30  
Majority Decision: 0.90  
Split Decision: 0.60  

Effective K:

`K_eff = K * M(fight)`

New update:

`R_1' = R_1 + K_eff * (S_1 - E_1)`
`R_2' = R_2 + K_eff * (S_2 - E_2)`

#### Prediction

Predict Fighter 1 wins if `R_1 > R_2`.

### K-Factor Optimization

Despite the filename, the script performs a grid search over K values.

#### Steps

1. Split historical data by date (80% train, 20% validation).
2. For K in a search range:
   - Compute Elo ratings
   - Evaluate validation accuracy
   - Select best K
3. Freeze ratings from training set
4. Test on `past3_events.csv` for genuine out-of-sample scoring

## Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy
