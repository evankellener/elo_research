"""
Example: Using Genetic Algorithm to Optimize Elo Parameters

This script demonstrates how to use the genetic algorithm optimization
to find the best Elo parameters for predicting MMA fight outcomes.

Usage:
    python examples/example_ga_optimization.py                    # Default: accuracy
    python examples/example_ga_optimization.py --optimize-for roi
    python examples/example_ga_optimization.py --optimize-for log_loss
    python examples/example_ga_optimization.py --optimize-for accuracy

Optimization Modes:
    - accuracy: Optimize for prediction accuracy (higher is better, 0-1)
    - roi: Optimize for return on investment (higher is better, -1 to 1)
    - log_loss: Optimize for logarithmic loss (lower raw values are better,
                but displayed as higher fitness for GA)
    - composite: Optimize for weighted combination of all metrics
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from optimization.ga_engine import GeneticAlgorithm, create_elo_fitness_function
from optimization.optimal_k_with_mov import add_bout_counts


def example_basic_optimization(optimize_for="accuracy"):
    """
    Basic GA optimization of K-factor and denominator.
    
    Args:
        optimize_for: Optimization target - "accuracy", "roi", "log_loss", or "composite"
    """
    print("="*70)
    print("EXAMPLE: BASIC GA OPTIMIZATION (K-factor + Denominator)")
    print("="*70)
    print(f"\nOptimization mode: {optimize_for}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    
    # Preprocess
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Use subset for faster demo
    df = df.tail(3000).copy()
    df = add_bout_counts(df)
    
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    print(f"Using {len(df)} fights for optimization")
    
    # Define parameter space
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600)
    }
    
    # Create fitness function
    fitness_fn = create_elo_fitness_function(
        df,
        base_elo=1500,
        use_validation_split=True,
        validation_percentile=0.8,
        optimize_for=optimize_for
    )
    
    # Create and run GA
    print("\nRunning genetic algorithm...")
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=20,
        elite_size=3,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        selection_method="tournament",
        crossover_method="uniform",
        random_seed=42,
        verbose=True
    )
    
    best_individual = ga.run(generations=20, early_stop_generations=10)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best K-factor: {best_individual.genes['k']:.2f}")
    print(f"Best denominator: {best_individual.genes['denominator']:.2f}")
    
    # Display the appropriate metric name based on optimization target
    metric_names = {
        "accuracy": "accuracy",
        "roi": "ROI",
        "log_loss": "log loss fitness",
        "composite": "composite fitness"
    }
    metric_name = metric_names.get(optimize_for, "fitness")
    print(f"Best {metric_name}: {best_individual.fitness:.4f}")
    
    # Add explanatory note for log_loss
    if optimize_for == "log_loss":
        print("\nNote: Log loss fitness is converted for GA optimization (exp(-log_loss)).")
        print("      Higher fitness values are better:")
        print("        1.0 = perfect predictions (log_loss=0)")
        print("        ~0.5 = random guessing (log_loss=0.693)")
        print("        <0.5 = worse than random")
    
    return best_individual, ga


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Optimize Elo parameters using Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/example_ga_optimization.py
  python examples/example_ga_optimization.py --optimize-for roi
  python examples/example_ga_optimization.py --optimize-for log_loss
  
Optimization Modes:
  accuracy  : Maximize prediction accuracy (0.0 to 1.0)
  roi       : Maximize return on investment (-1.0 to 1.0)
  log_loss  : Minimize logarithmic loss (shown as fitness 0.0 to 1.0)
  composite : Optimize weighted combination of all metrics
        """
    )
    parser.add_argument(
        "--optimize-for",
        type=str,
        default="accuracy",
        choices=["accuracy", "roi", "log_loss", "composite"],
        help="Optimization target (default: accuracy)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GENETIC ALGORITHM OPTIMIZATION EXAMPLE")
    print("="*70)
    
    # Check if data file exists
    if not os.path.exists("data/interleaved_cleaned.csv"):
        print("\nERROR: Data file not found!")
        print("Please ensure 'data/interleaved_cleaned.csv' exists.")
        sys.exit(1)
    
    # Run example
    best, ga = example_basic_optimization(optimize_for=args.optimize_for)
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
