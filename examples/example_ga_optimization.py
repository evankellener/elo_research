"""
Example: Using Genetic Algorithm to Optimize Elo Parameters

This script demonstrates how to use the genetic algorithm optimization
to find the best Elo parameters for predicting MMA fight outcomes.

Usage:
    python examples/example_ga_optimization.py                    # Default: accuracy
    python examples/example_ga_optimization.py --optimize-for roi
    python examples/example_ga_optimization.py --optimize-for log_loss
    python examples/example_ga_optimization.py --optimize-for accuracy
    python examples/example_ga_optimization.py --optimize-for roi --include-calibration

Optimization Modes:
    - accuracy: Optimize for prediction accuracy (higher is better, 0-1)
    - roi: Optimize for return on investment (displayed as percentage, -100% to 100%)
    - log_loss: Optimize for logarithmic loss (lower raw values are better,
                but displayed as higher fitness for GA)
    - composite: Optimize for weighted combination of all metrics

Calibration Metrics Flag:
    --include-calibration: When enabled, includes calibration metrics in the fitness
                          function to encourage better-calibrated predictions:
                          - ECE (Expected Calibration Error): Measures how well 
                            predicted probabilities match actual win rates
                          - Brier Score: Measures accuracy of probability predictions
                          - Sharpe Ratio: Risk-adjusted return measure from betting
                          
                          This helps find parameters that not only maximize the primary
                          metric (accuracy/ROI/log_loss) but also produce well-calibrated
                          probability predictions that may generalize better to 
                          out-of-sample data.

Output:
    The script saves two files after optimization:
    - ga_optimization_{mode}_{timestamp}.json: Complete results including best parameters,
                                                fitness scores, and generation history
    - ga_optimization_{mode}_{timestamp}_history.csv: Generation-by-generation history
                                                       for plotting and analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import json
from datetime import datetime
from optimization.ga_engine import GeneticAlgorithm, create_elo_fitness_function, MIN_BET_PROBABILITY
from optimization.optimal_k_with_mov import add_bout_counts, run_basic_elo, latest_ratings_from_trained_df
from elo.visualization import plot_ga_optimization_history


def _print_sample_events(df: pd.DataFrame, num_events: int = 5) -> None:
    """Print a handful of events and the fights from this training slice.

    This gives visibility into the concrete bouts the GA is optimizing over.
    """

    if "EVENT" not in df.columns or df.empty:
        print("\nNo EVENT column found; skipping event preview.")
        return

    # Reproducible sample of unique events from the filtered training data
    events_in_order = df.dropna(subset=["EVENT"])
    unique_events = events_in_order["EVENT"].drop_duplicates().tolist()

    if not unique_events:
        print("\nNo events available for preview after filtering.")
        return

    rng = np.random.default_rng(42)
    selected_events = (
        rng.choice(unique_events, size=min(num_events, len(unique_events)), replace=False)
        if len(unique_events) > num_events
        else unique_events
    )

    print("\nSample of events and fights used in this run:")
    for event_name in selected_events:
        fights = df[df["EVENT"] == event_name]
        print(f"\n  Event: {event_name} ({len(fights)} fights)")
        for _, row in fights.iterrows():
            date_str = "" if "DATE" not in row or pd.isna(row["DATE"]) else pd.to_datetime(row["DATE"]).date()
            fighter = row.get("FIGHTER", "?")
            opponent = row.get("opp_FIGHTER", "?")
            result = row.get("result", "?")
            print(f"    - {date_str}: {fighter} vs {opponent} (result: {result})")


def test_out_of_sample_metrics(
    df_trained_with_elo,
    test_df,
    denominator,
    base_elo=1500,
    confidence_threshold=50,
    optimize_for="accuracy",
    verbose=True,
    include_calibration=False
):
    """
    Test on out-of-sample data (past3_events.csv) and compute metrics.
    
    Args:
        df_trained_with_elo: Training DataFrame with Elo ratings already computed
        test_df: Test DataFrame (past3_events.csv)
        denominator: Denominator parameter for probability calculation
        base_elo: Base Elo rating for new fighters
        confidence_threshold: DEPRECATED - No longer used. Bets are placed on all valid fights.
                            Kept for backward compatibility.
        optimize_for: Metric to optimize for ("accuracy", "roi", "log_loss")
        verbose: Whether to print verbose output
        include_calibration: If True, compute and display calibration metrics
        
    Returns:
        Dictionary with metrics: accuracy, roi, log_loss, and optionally calibration metrics
    """
    # Get final ratings from training data
    rating_lookup = latest_ratings_from_trained_df(df_trained_with_elo, base_elo=base_elo)
    
    # Prepare test data
    tdf = test_df.copy()
    tdf["result"] = pd.to_numeric(tdf["result"], errors="coerce")
    tdf["date"] = pd.to_datetime(tdf["date"])
    
    predictions = []
    actuals = []
    
    for _, row in tdf.iterrows():
        f1 = row["fighter"]
        f2 = row["opp_fighter"]
        res = row["result"]
        
        if res not in (0, 1):
            continue
        
        # Use postcomp_elo from training as precomp_elo for OOS
        r1 = rating_lookup.get(f1, base_elo)
        r2 = rating_lookup.get(f2, base_elo)
        
        # Skip if ratings are equal
        if r1 == r2:
            continue
        
        # Calculate prediction probability
        elo_diff = r1 - r2
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / denominator))
        
        predictions.append(pred_prob)
        actuals.append(int(res))
    
    if len(predictions) == 0:
        return {
            'accuracy': 0.0,
            'roi': -1.0,
            'log_loss': float('inf'),
            'n_predictions': 0
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate accuracy
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_labels == actuals)
    
    # Calculate ROI - place bet on every valid fight (regardless of confidence threshold)
    total_profit = 0.0
    total_bets = 0
    for pred, actual in zip(predictions, actuals):
        # Determine predicted winner and their win probability
        pred_winner = 1 if pred > 0.5 else 0
        pred_prob = pred if pred > 0.5 else (1 - pred)
        
        # Calculate realistic payout based on implied odds
        pred_prob_clamped = max(pred_prob, MIN_BET_PROBABILITY)
        payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
        
        # We always bet 1 unit
        if pred_winner == actual:
            # Win: get back bet + payout
            total_profit += payout_multiplier
        else:
            # Lose: lose the bet
            total_profit -= 1.0
        total_bets += 1
    
    roi = (total_profit / total_bets) if total_bets > 0 else -1.0
    
    # Calculate log loss
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = np.mean(
        -(actuals * np.log(predictions_clipped) +
          (1 - actuals) * np.log(1 - predictions_clipped))
    )
    
    # Calculate calibration metrics if requested
    ece = None
    brier_score = None
    sharpe_ratio = None
    
    if include_calibration:
        # Calculate Brier Score
        brier_score = np.mean((predictions - actuals) ** 2)
        
        # Calculate ECE
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(predictions)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            if i == n_bins - 1:
                mask = (predictions >= bin_lower) & (predictions <= bin_upper)
            else:
                mask = (predictions >= bin_lower) & (predictions < bin_upper)
            
            bin_preds = predictions[mask]
            bin_actuals = actuals[mask]
            
            if len(bin_preds) > 0:
                avg_pred = np.mean(bin_preds)
                actual_rate = np.mean(bin_actuals)
                bin_error = abs(avg_pred - actual_rate)
                ece += bin_error * len(bin_preds) / total_samples
        
        # Calculate Sharpe Ratio from betting returns
        profits = []
        for pred, actual in zip(predictions, actuals):
            pred_winner = 1 if pred > 0.5 else 0
            pred_prob = pred if pred > 0.5 else (1 - pred)
            pred_prob_clamped = max(pred_prob, MIN_BET_PROBABILITY)
            payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
            
            if pred_winner == actual:
                profit = payout_multiplier
            else:
                profit = -1.0
            profits.append(profit)
        
        profits = np.array(profits)
        if len(profits) > 1 and np.std(profits) > 0:
            sharpe_ratio = np.mean(profits) / np.std(profits)
        else:
            sharpe_ratio = 0.0
    
    if verbose:
        print(f"\n{'='*70}")
        print("OUT-OF-SAMPLE TESTING (past3_events.csv)")
        print(f"{'='*70}")
        print(f"Number of predictions: {len(predictions)}")
        print(f"OOS Accuracy: {accuracy:.4f}")
        print(f"OOS ROI: {roi:.4f} ({roi * 100:.2f}%)")
        print(f"OOS Log Loss: {log_loss:.4f}")
        if include_calibration:
            print(f"OOS Brier Score: {brier_score:.4f}")
            print(f"OOS ECE: {ece:.4f}")
            print(f"OOS Sharpe Ratio: {sharpe_ratio:.4f}")
        if total_bets > 0:
            print(f"Number of bets placed: {total_bets}")
        print(f"{'='*70}")
    
    result = {
        'accuracy': accuracy,
        'roi': roi,
        'log_loss': log_loss,
        'n_predictions': len(predictions),
        'n_bets': total_bets
    }
    
    if include_calibration:
        result['brier_score'] = brier_score
        result['ece'] = ece
        result['sharpe_ratio'] = sharpe_ratio

    return result


# Prevent pytest from treating this helper as a collected test while keeping it importable
test_out_of_sample_metrics.__test__ = False


def example_basic_optimization(optimize_for="accuracy", include_calibration=False):
    """
    Basic GA optimization of K-factor and denominator.
    
    Args:
        optimize_for: Optimization target - "accuracy", "roi", "log_loss", or "composite"
        include_calibration: If True, includes calibration metrics (ECE, Brier, Sharpe) in fitness
    """
    print("="*70)
    print("EXAMPLE: BASIC GA OPTIMIZATION (K-factor + Denominator + MOV Weights)")
    print("="*70)
    print(f"\nOptimization mode: {optimize_for}")
    if include_calibration:
        print("Calibration metrics: ENABLED (ECE, Brier Score, Sharpe Ratio)")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("data/past3_events.csv", low_memory=False)
    
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
    print(f"Loaded {len(test_df)} fights for out-of-sample testing")

    _print_sample_events(df)
    
    # Define parameter space
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'w_ko': (0.8, 1.8),
        'w_sub': (0.8, 1.8),
        'w_udec': (0.7, 1.3),
        'w_mdec': (0.6, 1.2),
        'w_sdec': (0.4, 1.1),
    }
    
    # Create fitness function
    fitness_fn = create_elo_fitness_function(
        df,
        base_elo=1500,
        use_validation_split=True,
        validation_percentile=0.8,
        optimize_for=optimize_for,
        include_calibration=include_calibration
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
        verbose=True,
        optimization_mode=optimize_for
    )
    
    best_individual = ga.run(generations=20, early_stop_generations=10)
    
    # Recalculate Elo ratings with best parameters on full training data
    print("\nRecalculating Elo ratings with best parameters...")
    mov_param_keys = ["w_ko", "w_sub", "w_udec", "w_mdec", "w_sdec"]
    best_mov_params = {k: v for k, v in best_individual.genes.items() if k in mov_param_keys}
    use_mov = bool(best_mov_params)

    df_with_best_elo = run_basic_elo(
        df.copy(),
        k=best_individual.genes['k'],
        base_elo=1500,
        denominator=best_individual.genes['denominator'],
        use_mov=use_mov,
        mov_params=best_mov_params if use_mov else None
    )
    
    # Test on out-of-sample data
    oos_metrics = test_out_of_sample_metrics(
        df_with_best_elo,
        test_df,
        denominator=best_individual.genes['denominator'],
        base_elo=1500,
        confidence_threshold=best_individual.genes.get('confidence_threshold', 50),
        optimize_for=optimize_for,
        verbose=True,
        include_calibration=include_calibration
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best K-factor: {best_individual.genes['k']:.2f}")
    print(f"Best denominator: {best_individual.genes['denominator']:.2f}")
    if use_mov:
        print("Best MOV weights:")
        for key in mov_param_keys:
            print(f"  {key}: {best_individual.genes[key]:.3f}")
    
    # Calculate mode-specific metrics once
    log_loss_value = None
    roi_percent = None
    
    if optimize_for == "log_loss":
        fitness_clamped = max(best_individual.fitness, 1e-15)
        log_loss_value = -np.log(fitness_clamped)
    elif optimize_for == "roi":
        roi_percent = best_individual.fitness * 100
    
    # Display the appropriate metric based on optimization target
    if optimize_for == "log_loss":
        print(f"Best log loss: {log_loss_value:.4f}")
    elif optimize_for == "roi":
        print(f"Best ROI: {roi_percent:.2f}% (fitness: {best_individual.fitness:.4f})")
    else:
        metric_names = {
            "accuracy": "accuracy",
            "composite": "composite fitness"
        }
        metric_name = metric_names.get(optimize_for, "fitness")
        print(f"Best {metric_name}: {best_individual.fitness:.4f}")

    print("\nExplanation of results:")
    print("  • The GA is trained on a time-based validation split (last 20% of training data).")
    print("  • 'Best' values above reflect validation performance used for evolution.")
    print("  • The OUT-OF-SAMPLE section below reports metrics on past3_events.csv.")
    if optimize_for == "roi":
        print("  • ROI fitness is the average return per $1 bet on the validation set.")
        print("    Positive values mean profit, negative values mean loss; ±1.0 are hard bounds.")
    elif optimize_for == "log_loss":
        print("  • GA fitness internally uses exp(-log_loss); the printed log loss is the raw metric.")
    else:
        print("  • Accuracy/composite fitness values are bounded between 0 and 1.")
    
    # Add explanatory note for log_loss
    if optimize_for == "log_loss":
        print("\nNote: Log loss fitness is converted for GA optimization (exp(-log_loss)).")
        print("      Higher fitness values are better:")
        print("        1.0 = perfect predictions (log_loss=0)")
        print("        ~0.5 = random guessing (log_loss=0.693)")
        print("        <0.5 = worse than random")
    elif optimize_for == "roi":
        print("\nNote: ROI is displayed as a percentage of return on investment.")
        print("      For example, 34.92% means $1 bet returns $1.3492 on average.")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ga_optimization_{optimize_for}_{timestamp}.json"
    history_file = f"ga_optimization_{optimize_for}_{timestamp}_history.csv"
    
    # Prepare results dictionary
    results = {
        "timestamp": timestamp,
        "optimization_mode": optimize_for,
        "best_parameters": best_individual.genes,
        "best_fitness": best_individual.fitness,
        "generations": len(ga.history),
        "population_size": ga.population_size,
        "elite_size": ga.elite_size,
        "mutation_rate": ga.mutation_rate,
        "crossover_rate": ga.crossover_rate,
        "history": ga.history,
        "oos_metrics": oos_metrics
    }
    
    # Add mode-specific metrics
    if log_loss_value is not None:
        results["best_log_loss"] = log_loss_value
    if roi_percent is not None:
        results["best_roi_percent"] = roi_percent
    
    # Save results with error handling
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    except (IOError, OSError) as e:
        print(f"\nWarning: Could not save results to {output_file}: {e}")
    
    # Save history as CSV for easy plotting
    try:
        history_df = ga.get_history_dataframe()
        history_df.to_csv(history_file, index=False)
        print(f"History saved to: {history_file}")
    except (IOError, OSError) as e:
        print(f"Warning: Could not save history to {history_file}: {e}")
    
    # Generate and save visualization plots
    print("\nGenerating optimization visualization...")
    try:
        # Reuse history_df from above to avoid redundant computation
        plot_filename = f"ga_optimization_{optimize_for}_{timestamp}.png"
        plot_ga_optimization_history(
            history_df, 
            optimize_for=optimize_for,
            save_path=plot_filename,
            show_plot=True
        )
        print(f"Visualization saved to: {plot_filename}")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    # Provide a short interpretation guide for the printed numbers
    print("\nHow to read these results:")
    print("  • Best metrics shown above come from the validation slice used during GA search.")
    print("  • OOS metrics show generalization on past3_events.csv (not seen during training).")
    print("  • Accuracy: proportion of correct winners. 0.5 ≈ coin flip; above that is better.")
    print("  • ROI: average betting return per $1 when betting every fight.")
    print("    Positive ROI means profit; negative ROI means loss. Bounds are roughly -1.0 to +1.0.")
    print("  • Log loss: lower raw log-loss is better; fitness shown during GA is exp(-log_loss).")
    if include_calibration:
        print("  • Brier/ECE: lower means probabilities match reality better. Sharpe > 0 is preferable.")

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
  python examples/example_ga_optimization.py --optimize-for roi --include-calibration
  
Optimization Modes:
  accuracy  : Maximize prediction accuracy (0.0 to 1.0)
  roi       : Maximize return on investment (-1.0 to 1.0)
  log_loss  : Minimize logarithmic loss (shown as fitness 0.0 to 1.0)
  composite : Optimize weighted combination of all metrics

Calibration Metrics:
  --include-calibration : Includes ECE, Brier Score, and Sharpe Ratio in fitness.
                         This encourages well-calibrated probability predictions
                         that may generalize better to out-of-sample data.
        """
    )
    parser.add_argument(
        "--optimize-for",
        type=str,
        default="accuracy",
        choices=["accuracy", "roi", "log_loss", "composite"],
        help="Optimization target (default: accuracy)"
    )
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Include calibration metrics (ECE, Brier Score, Sharpe Ratio) in fitness function"
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
    best, ga = example_basic_optimization(
        optimize_for=args.optimize_for,
        include_calibration=args.include_calibration
    )
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
