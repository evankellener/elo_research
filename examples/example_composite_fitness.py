"""
Example: Using Composite Fitness Function

This script demonstrates how to use the composite fitness function with
different metric weights to optimize for different goals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from optimization.ga_engine import GeneticAlgorithm, create_elo_fitness_function
from optimization.optimal_k_with_mov import add_bout_counts

# Configuration
DATA_SIZE_FOR_EXAMPLES = 3000  # Use last N fights for faster demonstration


def example_balanced_optimization():
    """Example 1: Balanced optimization across all metrics."""
    print("="*70)
    print("EXAMPLE 1: BALANCED OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.tail(DATA_SIZE_FOR_EXAMPLES).copy()
    df = add_bout_counts(df)
    
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    print(f"Using {len(df)} fights for optimization")
    
    # Balanced weights - equal contribution from all metrics
    fitness_weights = {
        'accuracy': 0.25,
        'log_loss': 0.25,
        'brier_score': 0.25,
        'roi': 0.25
    }
    
    print("\nFitness weights (balanced):")
    for metric, weight in fitness_weights.items():
        print(f"  {metric}: {weight*100:.0f}%")
    
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'confidence_threshold': (30, 150)
    }
    
    fitness_fn = create_elo_fitness_function(
        df,
        optimize_for='composite',
        fitness_weights=fitness_weights
    )
    
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=20,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.8,
        random_seed=42,
        verbose=True
    )
    
    best = ga.run(generations=15)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("Best parameters:")
    for param, value in best.genes.items():
        print(f"  {param}: {value:.2f}")
    print(f"\nBest composite fitness: {best.fitness:.6f}")
    
    return best, ga


def example_accuracy_focused():
    """Example 2: Focus on accuracy and calibration (prediction quality)."""
    print("\n" + "="*70)
    print("EXAMPLE 2: ACCURACY-FOCUSED OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.tail(DATA_SIZE_FOR_EXAMPLES).copy()
    df = add_bout_counts(df)
    
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    # Weights emphasizing prediction accuracy and calibration
    fitness_weights = {
        'accuracy': 0.4,      # Higher weight on accuracy
        'log_loss': 0.3,      # Higher weight on calibration
        'brier_score': 0.3,   # Higher weight on calibration
        'roi': 0.0            # No ROI consideration
    }
    
    print("\nFitness weights (accuracy-focused):")
    for metric, weight in fitness_weights.items():
        print(f"  {metric}: {weight*100:.0f}%")
    
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600)
    }
    
    fitness_fn = create_elo_fitness_function(
        df,
        optimize_for='composite',
        fitness_weights=fitness_weights
    )
    
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=20,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.8,
        random_seed=43,
        verbose=True
    )
    
    best = ga.run(generations=15)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("Best parameters:")
    for param, value in best.genes.items():
        print(f"  {param}: {value:.2f}")
    print(f"\nBest composite fitness: {best.fitness:.6f}")
    
    return best, ga


def example_profit_focused():
    """Example 3: Focus on profitability (ROI and calibration)."""
    print("\n" + "="*70)
    print("EXAMPLE 3: PROFIT-FOCUSED OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df = df.tail(DATA_SIZE_FOR_EXAMPLES).copy()
    df = add_bout_counts(df)
    
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    # Weights emphasizing profitability
    fitness_weights = {
        'accuracy': 0.15,     # Lower weight on simple accuracy
        'log_loss': 0.2,      # Calibration still important
        'brier_score': 0.15,  # Calibration still important
        'roi': 0.5            # Highest weight on ROI
    }
    
    print("\nFitness weights (profit-focused):")
    for metric, weight in fitness_weights.items():
        print(f"  {metric}: {weight*100:.0f}%")
    
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'confidence_threshold': (30, 150)  # Important for ROI
    }
    
    fitness_fn = create_elo_fitness_function(
        df,
        optimize_for='composite',
        fitness_weights=fitness_weights
    )
    
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=20,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.8,
        random_seed=44,
        verbose=True
    )
    
    best = ga.run(generations=15)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print("Best parameters:")
    for param, value in best.genes.items():
        print(f"  {param}: {value:.2f}")
    print(f"\nBest composite fitness: {best.fitness:.6f}")
    
    return best, ga


if __name__ == "__main__":
    print("\n" + "="*70)
    print("COMPOSITE FITNESS FUNCTION EXAMPLES")
    print("="*70)
    print("\nThis script shows how to customize fitness weights for different goals:")
    print("1. Balanced: Equal weight to all metrics")
    print("2. Accuracy-focused: Emphasize prediction quality")
    print("3. Profit-focused: Emphasize ROI and betting returns")
    print("\n" + "="*70)
    
    # Check if data file exists
    if not os.path.exists("data/interleaved_cleaned.csv"):
        print("\nERROR: Data file not found!")
        print("Please ensure 'data/interleaved_cleaned.csv' exists.")
        sys.exit(1)
    
    # Run example 1
    best1, ga1 = example_balanced_optimization()
    
    # Uncomment to run additional examples:
    # best2, ga2 = example_accuracy_focused()
    # best3, ga3 = example_profit_focused()
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("- Adjust fitness_weights to match your optimization goals")
    print("- Balanced weights (25% each) work well for general use")
    print("- Accuracy-focused: Higher weights on accuracy + calibration metrics")
    print("- Profit-focused: Higher weight on ROI + moderate calibration")
    print("- Calibration metrics (log_loss, brier_score) help with probability estimates")
