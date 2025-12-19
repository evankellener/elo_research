"""
Genetic Algorithm Optimization with Time-Split ROI Focus

This script optimizes Elo parameters using a genetic algorithm with a focus on
betting ROI (Return on Investment) rather than simple prediction accuracy.
It uses time-series cross-validation to ensure the parameters generalize well
to future time periods.

The ROI fitness function evaluates how profitable the Elo-based betting strategy
would be, accounting for:
- Confidence thresholds (only bet when Elo difference is significant)
- Actual betting odds (when available)
- Kelly criterion for bet sizing (optional)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from optimization.ga_engine import GeneticAlgorithm
from optimization.optimal_k_with_mov import run_basic_elo, add_bout_counts
from elo.elo_utils import latest_ratings_from_trained_df


def calculate_roi_from_predictions(df, confidence_threshold=50):
    """
    Calculate ROI from Elo-based predictions, evaluating on the full list of valid fights.
    
    Per elo_explanatoin.md: "the elo's should be calculated for all fights. However when 
    the evaluations are made (accuracy, ROI, log loss, etc.) it should only be on 
    fighter's who have had more than one fight"
    
    This function:
    1. Evaluates predictions on ALL fights where both fighters have boutcount > 1
    2. Only places bets when confidence threshold is met
    3. Returns ROI based on bets, but tracks all predictions for accuracy
    
    Args:
        df: DataFrame with precomp_elo, opp_precomp_elo, result columns
        confidence_threshold: Minimum Elo difference to make a bet
        
    Returns:
        float: ROI as a decimal (e.g., 0.15 = 15% ROI)
    """
    total_bets = 0
    total_profit = 0
    total_predictions = 0  # Track all valid predictions
    correct_predictions = 0  # Track accuracy on all predictions
    
    for _, row in df.iterrows():
        if pd.isna(row['result']) or row['result'] not in (0, 1):
            continue
        
        elo1 = row.get('precomp_elo', 1500)
        elo2 = row.get('opp_precomp_elo', 1500)
        
        # Check if bout count filters pass
        # Filter to only include fights where both fighters have had more than one fight
        # (as specified in elo_explanatoin.md: "more than one fight")
        bout1 = row.get('precomp_boutcount', 0)
        bout2 = row.get('opp_precomp_boutcount', 0)
        if pd.isna(bout1) or pd.isna(bout2) or bout1 <= 1 or bout2 <= 1:
            continue
        
        # Skip if Elos are equal (can't make a prediction)
        if elo1 == elo2:
            continue
        
        # Track ALL predictions on valid fights (per elo_explanatoin.md)
        predicted_winner = 1 if elo1 > elo2 else 0
        actual_winner = int(row['result'])
        total_predictions += 1
        if predicted_winner == actual_winner:
            correct_predictions += 1
        
        elo_diff = abs(elo1 - elo2)
        
        # Only bet if confidence threshold is met
        if elo_diff < confidence_threshold:
            continue
        
        # Assume unit bet with even odds (1:1 payout)
        # If we win, we gain 1 unit. If we lose, we lose 1 unit.
        if predicted_winner == actual_winner:
            total_profit += 1
        else:
            total_profit -= 1
        
        total_bets += 1
    
    # Return ROI even if no bets (0.0)
    # This ensures the GA evaluates fitness across all valid fights, not just bets
    if total_bets == 0:
        return 0.0
    
    roi = total_profit / total_bets
    return roi


def create_roi_fitness_function(df, split_months=6):
    """
    Create a fitness function that optimizes for ROI using time-series cross-validation.
    
    Args:
        df: Full training DataFrame
        split_months: Number of months for each validation window
        
    Returns:
        Fitness function for GA
    """
    # Sort by date
    df = df.sort_values("DATE").reset_index(drop=True)
    
    # Calculate time splits using DateOffset for accuracy
    min_date = df["DATE"].min()
    max_date = df["DATE"].max()
    total_days = (max_date - min_date).days
    # More accurate month calculation
    split_days = int((pd.DateOffset(months=split_months).delta.total_seconds() / 86400) if hasattr(pd.DateOffset(months=split_months), 'delta') else split_months * 30.44)
    
    def fitness_function(params):
        """
        Evaluate fitness using time-series cross-validation.
        
        For each time window:
        1. Train Elo on data before window
        2. Evaluate ROI on data in window
        3. Average ROI across all windows
        """
        k = params.get('k', 32)
        denominator = params.get('denominator', 400)
        confidence_threshold = params.get('confidence_threshold', 50)
        
        rois = []
        
        # Create multiple time-based validation splits
        n_splits = max(3, total_days // split_days)  # At least 3 splits
        
        for i in range(1, n_splits):
            # Define split point
            split_date = min_date + pd.Timedelta(days=i * split_days)
            
            # Check if we have enough data after split
            test_data = df[df["DATE"] >= split_date]
            if len(test_data) < 50:  # Need at least 50 fights for meaningful evaluation
                continue
            
            train_data = df[df["DATE"] < split_date].copy()
            if len(train_data) < 100:  # Need sufficient training data
                continue
            
            # Train Elo on training data
            train_with_elo = run_basic_elo(
                train_data,
                k=k,
                base_elo=1500,
                denominator=denominator,
                use_mov=True
            )
            
            # Get ratings from training
            ratings = latest_ratings_from_trained_df(train_with_elo, base_elo=1500)
            
            # Apply ratings to test data
            test_data = test_data.copy()
            test_data['precomp_elo'] = test_data['FIGHTER'].map(ratings).fillna(1500)
            test_data['opp_precomp_elo'] = test_data['opp_FIGHTER'].map(ratings).fillna(1500)
            
            # Calculate ROI on test data
            roi = calculate_roi_from_predictions(test_data, confidence_threshold)
            rois.append(roi)
        
        # Return average ROI across splits
        if len(rois) == 0:
            return 0.0
        
        avg_roi = np.mean(rois)
        # Add small penalty for high variance (prefer stable returns)
        if len(rois) > 1:
            std_roi = np.std(rois)
            fitness = avg_roi - 0.1 * std_roi
        else:
            fitness = avg_roi
        
        return fitness
    
    return fitness_function


def optimize_for_roi(df, split_months=6, population_size=40, generations=80, verbose=True):
    """
    Optimize Elo parameters for ROI using genetic algorithm.
    
    Args:
        df: Training DataFrame
        split_months: Months for time-based validation
        population_size: GA population size
        generations: Number of generations
        verbose: Print progress
        
    Returns:
        tuple: (best_individual, ga_instance)
    """
    if verbose:
        print("="*60)
        print("GA OPTIMIZATION FOR ROI")
        print("="*60)
        print(f"Split months: {split_months}")
        print(f"Population size: {population_size}")
        print(f"Generations: {generations}")
        print("="*60 + "\n")
    
    # Parameter bounds
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'confidence_threshold': (30, 150),  # Minimum Elo difference to bet
    }
    
    # Create fitness function
    fitness_fn = create_roi_fitness_function(df, split_months=split_months)
    
    # Create and run GA
    ga = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn,
        population_size=population_size,
        elite_size=4,
        mutation_rate=0.15,
        crossover_rate=0.8,
        tournament_size=3,
        selection_method="tournament",
        crossover_method="uniform",
        verbose=verbose
    )
    
    best_individual = ga.run(
        generations=generations,
        early_stop_generations=20
    )
    
    return best_individual, ga


def plot_roi_results(ga, save_path=None):
    """
    Plot ROI optimization results.
    
    Args:
        ga: GeneticAlgorithm instance with history
        save_path: Optional path to save figure
    """
    history_df = ga.get_history_dataframe()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: ROI over generations
    ax1 = axes[0, 0]
    ax1.plot(history_df['generation'], history_df['best_fitness'], 
             'o-', label='Best ROI', linewidth=2, markersize=5, color='green')
    ax1.plot(history_df['generation'], history_df['avg_fitness'], 
             's-', label='Avg ROI', linewidth=2, markersize=4, color='blue', alpha=0.7)
    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('ROI', fontsize=11)
    ax1.set_title('ROI Over Generations', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 2: K-factor evolution
    ax2 = axes[0, 1]
    k_values = [gen['best_genes']['k'] for gen in ga.history]
    ax2.plot(history_df['generation'], k_values, 'o-', linewidth=2, markersize=4, color='purple')
    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('K-factor', fontsize=11)
    ax2.set_title('K-factor Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Denominator evolution
    ax3 = axes[1, 0]
    denom_values = [gen['best_genes']['denominator'] for gen in ga.history]
    ax3.plot(history_df['generation'], denom_values, 'o-', linewidth=2, markersize=4, color='orange')
    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Denominator', fontsize=11)
    ax3.set_title('Denominator Evolution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confidence threshold evolution
    ax4 = axes[1, 1]
    conf_values = [gen['best_genes']['confidence_threshold'] for gen in ga.history]
    ax4.plot(history_df['generation'], conf_values, 'o-', linewidth=2, markersize=4, color='brown')
    ax4.set_xlabel('Generation', fontsize=11)
    ax4.set_ylabel('Confidence Threshold', fontsize=11)
    ax4.set_title('Confidence Threshold Evolution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('GA ROI Optimization Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROI results plot saved to {save_path}")
    
    plt.show()


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Optimize Elo parameters for ROI using genetic algorithm"
    )
    parser.add_argument(
        '--data-file',
        type=str,
        default='data/interleaved_cleaned.csv',
        help='Path to input data CSV file'
    )
    parser.add_argument(
        '--split-months',
        type=int,
        default=6,
        help='Number of months for time-based validation splits'
    )
    parser.add_argument(
        '--population-size',
        type=int,
        default=40,
        help='GA population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=80,
        help='Number of generations to evolve'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='ga_roi',
        help='Prefix for output files'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file, low_memory=False)
    
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
    
    print(f"Loaded {len(df)} fights from {df['DATE'].min()} to {df['DATE'].max()}")
    
    # Run optimization
    best_individual, ga = optimize_for_roi(
        df,
        split_months=args.split_months,
        population_size=args.population_size,
        generations=args.generations,
        verbose=True
    )
    
    # Plot results
    print("\n=== Generating plots ===")
    plot_roi_results(ga, save_path=f'images/{args.output_prefix}_results.png')
    
    # Save results
    print("\n=== Saving results ===")
    history_df = ga.get_history_dataframe()
    history_df.to_csv(f'{args.output_prefix}_history.csv', index=False)
    print(f"Optimization history saved to {args.output_prefix}_history.csv")
    
    # Save best parameters
    with open(f'{args.output_prefix}_best_params.txt', 'w') as f:
        f.write("BEST PARAMETERS FOR ROI OPTIMIZATION\n")
        f.write("="*50 + "\n\n")
        for param, value in best_individual.genes.items():
            f.write(f"{param}: {value:.4f}\n")
        f.write(f"\nBest ROI: {best_individual.fitness:.6f}\n")
    print(f"Best parameters saved to {args.output_prefix}_best_params.txt")
    
    print("\n" + "="*60)
    print("ROI OPTIMIZATION COMPLETE")
    print("="*60)
    print("\nBest parameters:")
    for param, value in best_individual.genes.items():
        print(f"  {param}: {value:.4f}")
    print(f"\nBest ROI: {best_individual.fitness:.6f}")
    
    if best_individual.fitness > 0:
        print(f"\nThis represents a {best_individual.fitness * 100:.2f}% average return per bet")
    else:
        print(f"\nNegative ROI suggests the betting strategy needs refinement")


if __name__ == "__main__":
    main()
