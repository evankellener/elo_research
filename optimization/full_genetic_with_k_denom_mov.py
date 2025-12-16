"""
Full Genetic Algorithm Optimization for Elo Parameters

This script uses a true genetic algorithm (not grid search) to optimize multiple
Elo parameters simultaneously:
- K-factor: Controls how much ratings change after each fight
- Denominator: Controls rating difference sensitivity (standard is 400)
- MOV weights: Five weights for different fight outcomes (KO, Sub, UD, MD, SD)

The GA uses evolutionary operators (selection, crossover, mutation) to search
the parameter space efficiently, exploring combinations that grid search would miss.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimization.ga_engine import GeneticAlgorithm, create_elo_fitness_function
from elo.elo_utils import method_of_victory_scale, build_fighter_history, has_prior_history, add_bout_counts


def run_basic_elo_with_custom_mov(df, k=32, base_elo=1500, denominator=400, 
                                   mov_weights=None, use_mov=True, draw_k_factor=0.5):
    """
    Core Elo loop with customizable method of victory weights.
    
    Args:
        df: DataFrame with fight data
        k: K-factor
        base_elo: Starting Elo rating
        denominator: Denominator in Elo formula
        mov_weights: Dict with keys w_ko, w_sub, w_udec, w_mdec, w_sdec
        use_mov: If True, applies method of victory scaling
        draw_k_factor: Multiplier for K-factor in draws
    
    Returns:
        DataFrame with Elo ratings computed
    """
    from optimization.optimal_k_with_mov import run_basic_elo
    
    # For now, use the standard implementation
    # A more advanced version would modify method_of_victory_scale to use custom weights
    return run_basic_elo(df, k=k, base_elo=base_elo, denominator=denominator, 
                        use_mov=use_mov, draw_k_factor=draw_k_factor)


def optimize_elo_parameters_with_ga(
    df,
    param_bounds=None,
    population_size=50,
    generations=100,
    elite_size=5,
    mutation_rate=0.1,
    crossover_rate=0.8,
    early_stop_generations=15,
    optimize_for="composite",
    fitness_weights=None,
    verbose=True
):
    """
    Optimize Elo parameters using genetic algorithm with multiple metrics.
    
    Args:
        df: Training DataFrame with fight data
        param_bounds: Dict of parameter bounds (min, max). If None, uses defaults.
        population_size: Size of GA population
        generations: Number of generations to evolve
        elite_size: Number of elite individuals to preserve
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        early_stop_generations: Stop if no improvement for this many generations
        optimize_for: "accuracy", "roi", or "composite" (default)
        fitness_weights: Optional dict for composite fitness. Keys: 
            'accuracy', 'log_loss', 'brier_score', 'roi'
            Default: {'accuracy': 0.3, 'log_loss': 0.2, 'brier_score': 0.2, 'roi': 0.3}
        verbose: Print progress
        
    Returns:
        tuple: (best_individual, ga_instance)
    """
    # Default parameter bounds if not specified
    if param_bounds is None:
        param_bounds = {
            'k': (10, 500),
            'denominator': (200, 600),
            # Confidence threshold: minimum Elo difference (in scaled prob units) to place bet
            # Range 30-150 maps to ~0.03-0.15 probability difference from 0.5
            'confidence_threshold': (30, 150),
            'w_ko': (1.0, 2.0),
            'w_sub': (1.0, 2.0),
            'w_udec': (0.8, 1.5),
            'w_mdec': (0.7, 1.3),
            'w_sdec': (0.5, 1.2),
        }
    
    if verbose:
        print("="*60)
        print("GENETIC ALGORITHM OPTIMIZATION FOR ELO PARAMETERS")
        print("="*60)
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print(f"Elite size: {elite_size}")
        print(f"Mutation rate: {mutation_rate}")
        print(f"Crossover rate: {crossover_rate}")
        print(f"Optimization mode: {optimize_for}")
        print(f"Parameters to optimize: {list(param_bounds.keys())}")
        if fitness_weights:
            print(f"Fitness weights: {fitness_weights}")
        print("="*60 + "\n")
    
    # Create fitness function with composite metrics
    fitness_fn = create_elo_fitness_function(
        df,
        base_elo=1500,
        use_validation_split=True,
        validation_percentile=0.8,
        optimize_for=optimize_for,
        fitness_weights=fitness_weights
    )
    
    # Create and run GA
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=3,
        selection_method="tournament",
        crossover_method="uniform",
        verbose=verbose
    )
    
    best_individual = ga.run(
        generations=generations,
        early_stop_generations=early_stop_generations
    )
    
    return best_individual, ga


def plot_ga_convergence(ga, save_path=None):
    """
    Plot the convergence of the genetic algorithm.
    
    Args:
        ga: GeneticAlgorithm instance with history
        save_path: Optional path to save the figure
    """
    history_df = ga.get_history_dataframe()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fitness over generations
    ax1 = axes[0]
    ax1.plot(history_df['generation'], history_df['best_fitness'], 
             'o-', label='Best Fitness', linewidth=2, markersize=4, color='green')
    ax1.plot(history_df['generation'], history_df['avg_fitness'], 
             's-', label='Avg Fitness', linewidth=2, markersize=3, color='blue', alpha=0.7)
    ax1.plot(history_df['generation'], history_df['worst_fitness'], 
             '^-', label='Worst Fitness', linewidth=2, markersize=3, color='red', alpha=0.5)
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness (Accuracy)', fontsize=12)
    ax1.set_title('GA Convergence: Fitness over Generations', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitness improvement (zoomed on best)
    ax2 = axes[1]
    ax2.plot(history_df['generation'], history_df['best_fitness'], 
             'o-', linewidth=2, markersize=5, color='green')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Best Fitness (Accuracy)', fontsize=12)
    ax2.set_title('Best Fitness Progression', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()


def plot_parameter_evolution(ga, save_path=None):
    """
    Plot how parameters evolved over generations.
    
    Args:
        ga: GeneticAlgorithm instance with history
        save_path: Optional path to save the figure
    """
    history_df = ga.get_history_dataframe()
    
    # Extract parameter values from best_genes in history
    params = list(ga.param_bounds.keys())
    n_params = len(params)
    
    # Create subplots
    n_rows = (n_params + 2) // 3  # 3 columns
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes.flatten()
    
    for i, param in enumerate(params):
        ax = axes[i]
        values = [gen['best_genes'][param] for gen in ga.history]
        ax.plot(history_df['generation'], values, 'o-', linewidth=2, markersize=4)
        ax.set_xlabel('Generation', fontsize=11)
        ax.set_ylabel(param, fontsize=11)
        ax.set_title(f'{param} Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at final value
        final_value = values[-1]
        ax.axhline(y=final_value, color='r', linestyle='--', alpha=0.5, 
                   label=f'Final: {final_value:.3f}')
        ax.legend(loc='best', fontsize=9)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Parameter Evolution Over Generations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Parameter evolution plot saved to {save_path}")
    
    plt.show()


def compare_ga_vs_grid_search(df, test_df, ga_params, grid_k_range, verbose=True):
    """
    Compare GA-optimized parameters against grid search over K.
    
    Args:
        df: Training DataFrame
        test_df: Test DataFrame for OOS evaluation
        ga_params: Dict of parameters from GA optimization
        grid_k_range: Range of K values for grid search
        verbose: Print comparison results
        
    Returns:
        Dict with comparison results
    """
    from optimization.optimal_k_with_mov import (
        genetic_algorithm_k, run_basic_elo, 
        test_out_of_sample_accuracy
    )
    
    if verbose:
        print("\n" + "="*60)
        print("COMPARISON: GA vs GRID SEARCH")
        print("="*60)
    
    # Grid search with MOV
    if verbose:
        print("\nRunning grid search...")
    best_k_grid, best_acc_grid, _ = genetic_algorithm_k(
        df, k_range=grid_k_range, use_mov=True, 
        return_all_results=True, verbose=False
    )
    
    # Train with best grid K and evaluate OOS
    df_trained_grid = run_basic_elo(df.copy(), k=best_k_grid, use_mov=True)
    oos_acc_grid = test_out_of_sample_accuracy(
        df_trained_grid, test_df, best_k=best_k_grid, verbose=False
    )
    
    if verbose:
        print(f"Grid Search Results:")
        print(f"  Best K: {best_k_grid}")
        print(f"  Best validation accuracy: {best_acc_grid:.4f}")
        print(f"  OOS accuracy: {oos_acc_grid:.4f}")
    
    # GA optimization
    if verbose:
        print(f"\nGA Results:")
        print(f"  Parameters: {ga_params}")
    
    # Train with GA parameters and evaluate
    k_ga = ga_params.get('k', 32)
    denom_ga = ga_params.get('denominator', 400)
    df_trained_ga = run_basic_elo(
        df.copy(), k=k_ga, base_elo=1500, 
        denominator=denom_ga, use_mov=True
    )
    
    # Evaluate on validation set
    cutoff = df["DATE"].quantile(0.8)
    from optimization.optimal_k_with_mov import elo_accuracy
    acc_all_ga, acc_future_ga, _ = elo_accuracy(df_trained_ga, cutoff)
    
    # Evaluate OOS
    oos_acc_ga = test_out_of_sample_accuracy(
        df_trained_ga, test_df, best_k=k_ga, verbose=False
    )
    
    if verbose:
        print(f"  Validation accuracy: {acc_future_ga:.4f}")
        print(f"  OOS accuracy: {oos_acc_ga:.4f}")
        
        print(f"\nImprovement:")
        print(f"  Validation: {acc_future_ga - best_acc_grid:+.4f} ({(acc_future_ga - best_acc_grid) / best_acc_grid * 100:+.2f}%)")
        print(f"  OOS: {oos_acc_ga - oos_acc_grid:+.4f} ({(oos_acc_ga - oos_acc_grid) / oos_acc_grid * 100:+.2f}%)")
    
    return {
        'grid_k': best_k_grid,
        'grid_val_acc': best_acc_grid,
        'grid_oos_acc': oos_acc_grid,
        'ga_params': ga_params,
        'ga_val_acc': acc_future_ga,
        'ga_oos_acc': oos_acc_ga
    }


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("data/past3_events.csv", low_memory=False)
    
    # Preprocess
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE").reset_index(drop=True)
    df = add_bout_counts(df)
    
    # Ensure boutcount columns are numeric
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    # Define parameter space for GA
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'confidence_threshold': (30, 150),  # For ROI calculation
        'w_ko': (1.0, 2.0),
        'w_sub': (1.0, 2.0),
        'w_udec': (0.8, 1.5),
        'w_mdec': (0.7, 1.3),
        'w_sdec': (0.5, 1.2),
    }
    
    # Define fitness weights for composite optimization
    # These weights determine how much each metric contributes to the fitness score
    fitness_weights = {
        'accuracy': 0.3,       # Prediction accuracy (30%)
        'log_loss': 0.2,       # Log loss / cross-entropy (20%)
        'brier_score': 0.2,    # Brier score (20%)
        'roi': 0.3             # Return on investment (30%)
    }
    
    print("Fitness function uses composite metrics:")
    print(f"  - Accuracy: {fitness_weights['accuracy']*100:.0f}%")
    print(f"  - Log Loss: {fitness_weights['log_loss']*100:.0f}%")
    print(f"  - Brier Score: {fitness_weights['brier_score']*100:.0f}%")
    print(f"  - ROI: {fitness_weights['roi']*100:.0f}%")
    print()
    
    # Run GA optimization with composite metrics
    best_individual, ga = optimize_elo_parameters_with_ga(
        df,
        param_bounds=param_bounds,
        population_size=50,
        generations=100,
        elite_size=5,
        mutation_rate=0.15,
        crossover_rate=0.8,
        early_stop_generations=15,
        optimize_for="composite",
        fitness_weights=fitness_weights,
        verbose=True
    )
    
    # Plot convergence
    print("\n=== Generating convergence plots ===")
    plot_ga_convergence(ga, save_path='images/ga_convergence.png')
    plot_parameter_evolution(ga, save_path='images/ga_parameter_evolution.png')
    
    # Compare with grid search
    comparison = compare_ga_vs_grid_search(
        df, test_df,
        ga_params=best_individual.genes,
        grid_k_range=range(10, 500, 20),
        verbose=True
    )
    
    # Save results
    print("\n=== Saving results ===")
    results_df = ga.get_history_dataframe()
    results_df.to_csv('ga_optimization_history.csv', index=False)
    print("Optimization history saved to ga_optimization_history.csv")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("\nBest parameters found:")
    for param, value in best_individual.genes.items():
        print(f"  {param}: {value:.4f}")
    print(f"\nBest composite fitness score: {best_individual.fitness:.6f}")
    print("(Composite of: accuracy, log loss, Brier score, and ROI)")
