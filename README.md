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
│   ├── full_genetic_with_k_denom_mov.py
│   ├── ga_time_split_roi.py
│   └── optimal_k_with_mov.py
├── analysis/            # Analysis and diagnostic scripts
│   ├── analyze_baseline_diagnostics.py
│   ├── diagnostic_tests.py
│   └── prediction_metrics.py
├── tests/               # Test scripts
├── data/                # Fight data
└── images/              # Output images
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

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

## Optimization

Run genetic algorithm to find optimal parameters:

```bash
python optimization/full_genetic_with_k_denom_mov.py
```

Run time-split ROI optimization:

```bash
python optimization/ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 6
```

## Elo System

The Elo rating system assigns each fighter a numerical rating. Expected win probability:

E₁ = 1 / (1 + 10^((R₂ - R₁) / 400))

Rating update after fight:

R₁' = R₁ + K × (S₁ - E₁)

Where K is the K-factor and S is the actual result (1 for win, 0 for loss).

## Method of Victory

MOV weights scale the K-factor based on fight outcome:
- KO/TKO: 1.4x
- Submission: 1.3x
- Unanimous Decision: 1.0x
- Majority Decision: 0.9x
- Split Decision: 0.7x

This reflects that more decisive victories should have larger rating impacts.
