"""
Example: Using Genetic Algorithm to Optimize Elo Parameters

This script demonstrates how to use the genetic algorithm optimization
to find the best Elo parameters for predicting MMA fight outcomes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from optimization.ga_engine import GeneticAlgorithm, create_elo_fitness_function
from optimization.optimal_k_with_mov import add_bout_counts


def example_basic_optimization():
    """Basic GA optimization of K-factor and denominator."""
    print("="*70)
    print("EXAMPLE: BASIC GA OPTIMIZATION (K-factor + Denominator)")
    print("="*70)
    
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
        optimize_for="accuracy"
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
    print(f"Best accuracy: {best_individual.fitness:.4f}")
    
    return best_individual, ga


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GENETIC ALGORITHM OPTIMIZATION EXAMPLE")
    print("="*70)
    
    # Check if data file exists
    if not os.path.exists("data/interleaved_cleaned.csv"):
        print("\nERROR: Data file not found!")
        print("Please ensure 'data/interleaved_cleaned.csv' exists.")
        sys.exit(1)
    
    # Run example
    best, ga = example_basic_optimization()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
