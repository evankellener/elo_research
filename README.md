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

## Requirements

- Python 3.7+
- pandas
- matplotlib
- numpy

