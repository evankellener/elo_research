#!/usr/bin/env python3
"""
Time-split ROI optimization for GA-optimized ELO ranking.

This script implements a reproducible time-based validation pipeline that modifies
the genetic algorithm optimization objective to find parameter sets that produce
consistent ROI across time splits.

The composite fitness function is:
    fitness = mean_roi - lambda * std_roi

where:
- mean_roi: Arithmetic mean ROI across all time splits
- std_roi: Standard deviation of ROI across time splits
- lambda: Tunable penalty hyperparameter (default: 1.0)

This pushes the GA to prefer parameter sets with consistent ROI across time,
rather than overfitting to specific time periods.

Usage:
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv \\
        --split-months 6 --generations 30 --population 30

Alternative objective (coefficient of variation):
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv \\
        --split-months 6 --objective cv
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from time_splitter import create_time_splits, detect_time_column, get_split_info, filter_and_sample_splits
from full_genetic_with_k_denom_mov import (
    run_basic_elo,
    evaluate_params_roi,
    random_params,
    crossover,
    mutate,
    tournament_select,
    get_param_bounds,
    build_bidirectional_odds_lookup,
    american_odds_to_decimal,
)
from elo_utils import add_bout_counts, build_fighter_history, has_prior_history

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def compute_split_roi(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    params: Dict[str, float],
    time_column: str = 'DATE'
) -> Dict[str, Any]:
    """
    Compute ROI on a validation split using Elo model trained on training data.

    Args:
        train_df: Training data (all fights before validation period)
        val_df: Validation data (fights in the validation period)
        odds_df: Odds data for ROI calculation
        params: ELO hyperparameters (k, w_ko, w_sub, w_udec, w_sdec, w_mdec)
        time_column: Name of the date column

    Returns:
        dict with:
        - roi_percent: ROI percentage on validation data
        - num_bets: Number of bets placed
        - total_wagered: Total amount wagered
        - total_profit: Total profit/loss
        - wins: Number of winning bets
        - val_start: Start date of validation period
        - val_end: End date of validation period
    """
    # Ensure DATE column exists
    if time_column != 'DATE' and 'DATE' not in train_df.columns:
        train_df = train_df.rename(columns={time_column: 'DATE'})
        val_df = val_df.rename(columns={time_column: 'DATE'})

    # Ensure data is sorted by date
    train_df = train_df.sort_values('DATE').reset_index(drop=True)
    val_df = val_df.sort_values('DATE').reset_index(drop=True)

    # Build MOV params dict
    mov_params = {
        "w_ko": params["w_ko"],
        "w_sub": params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }

    # Run ELO on training data
    df_trained = run_basic_elo(train_df, k=params["k"], mov_params=mov_params)

    # Get fighter ratings at end of training
    ratings = {}
    for _, row in df_trained.iterrows():
        ratings[row["FIGHTER"]] = row["postcomp_elo"]
        ratings[row["opp_FIGHTER"]] = row["opp_postcomp_elo"]

    # Build fighter history for prior fight check
    hist = build_fighter_history(df_trained)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()

    # Build odds lookup
    odds_lookup = build_bidirectional_odds_lookup(odds_df)

    # Calculate ROI on validation data
    total_wagered = 0.0
    total_profit = 0.0
    num_bets = 0
    wins = 0
    processed_fights = set()

    for _, row in val_df.iterrows():
        result = row.get("result")
        if result not in (0, 1):
            continue

        fighter = row.get("FIGHTER")
        opponent = row.get("opp_FIGHTER")
        fight_date = row.get("DATE")

        if pd.isna(fight_date):
            continue

        # Create unique fight key
        date_str = str(pd.to_datetime(fight_date).date())
        fight_key = tuple(sorted([fighter, opponent])) + (date_str,)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)

        # Get ratings (use base 1500 for unknown fighters)
        r1 = ratings.get(fighter, 1500)
        r2 = ratings.get(opponent, 1500)

        # Skip if equal ratings
        if r1 == r2:
            continue

        # Skip if no prior history (can't reliably predict)
        if not has_prior_history(first_dates, fighter, fight_date):
            continue
        if not has_prior_history(first_dates, opponent, fight_date):
            continue

        # Determine higher ELO fighter (who we bet on)
        if r1 > r2:
            bet_on = fighter
            bet_against = opponent
            bet_won = (result == 1)
            odds_key = (fighter, opponent, date_str)
            alt_key = (opponent, fighter, date_str)
        else:
            bet_on = opponent
            bet_against = fighter
            bet_won = (result == 0)
            odds_key = (opponent, fighter, date_str)
            alt_key = (fighter, opponent, date_str)

        # Look up odds
        if odds_key in odds_lookup:
            bet_odds = odds_lookup[odds_key]
        elif alt_key in odds_lookup:
            bet_odds = odds_lookup[alt_key]
        else:
            continue  # No odds available

        decimal_odds = american_odds_to_decimal(bet_odds)
        if decimal_odds is None:
            continue

        # Simulate bet
        bet_amount = 1.0
        total_wagered += bet_amount
        num_bets += 1

        if bet_won:
            payout = bet_amount * decimal_odds
            profit = payout - bet_amount
            wins += 1
        else:
            profit = -bet_amount

        total_profit += profit

    # Calculate ROI
    roi_percent = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0.0

    return {
        'roi_percent': roi_percent,
        'num_bets': num_bets,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'wins': wins,
        'val_start': val_df['DATE'].min(),
        'val_end': val_df['DATE'].max(),
    }


def compute_time_split_fitness(
    df: pd.DataFrame,
    odds_df: pd.DataFrame,
    params: Dict[str, float],
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    lambda_penalty: float = 1.0,
    objective: str = "mean_std",
    time_column: str = 'DATE'
) -> Dict[str, Any]:
    """
    Compute fitness using time-split ROI consistency.

    The fitness function penalizes variance across time splits to find
    parameter sets that generalize well across different time periods.

    Args:
        df: Full historical data
        odds_df: Odds data
        params: ELO hyperparameters
        splits: List of (train_df, val_df) tuples from create_time_splits()
        lambda_penalty: Penalty weight for std deviation (default 1.0)
        objective: Fitness objective:
            - "mean_std": fitness = mean_roi - lambda * std_roi (default)
            - "cv": fitness = -coefficient_of_variation (minimize CV)
        time_column: Name of the date column

    Returns:
        dict with:
        - fitness: Composite fitness score (higher is better)
        - mean_roi: Mean ROI across splits
        - std_roi: Standard deviation of ROI across splits
        - cv: Coefficient of variation (std/|mean|)
        - per_split_roi: List of ROI values for each split
        - per_split_details: List of dicts with details for each split
    """
    per_split_roi = []
    per_split_details = []

    for train_df, val_df in splits:
        result = compute_split_roi(train_df, val_df, odds_df, params, time_column)
        per_split_roi.append(result['roi_percent'])
        per_split_details.append(result)

    # Calculate statistics
    roi_array = np.array(per_split_roi)
    mean_roi = float(np.mean(roi_array))
    std_roi = float(np.std(roi_array, ddof=1)) if len(roi_array) > 1 else 0.0

    # Coefficient of variation (handle zero mean)
    # Use a large finite value instead of inf to avoid arithmetic issues
    MAX_CV = 1e6  # Large but finite value
    if abs(mean_roi) > 0.001:
        cv = std_roi / abs(mean_roi)
    else:
        cv = MAX_CV if std_roi > 0 else 0.0

    # Calculate fitness based on objective
    if objective == "mean_std":
        # Maximize: mean_roi - lambda * std_roi
        # Higher mean ROI is good, lower variance is good
        fitness = mean_roi - lambda_penalty * std_roi
    elif objective == "cv":
        # Minimize coefficient of variation (negate for maximization)
        # We want low CV, so fitness = -CV (or use a transform)
        # Clamp CV to reasonable range for fitness calculation
        clamped_cv = min(cv, 100.0)  # Cap at 100 for reasonable fitness scale
        fitness = 100.0 - clamped_cv * 100.0
    else:
        raise ValueError(f"Unknown objective: {objective}. Use 'mean_std' or 'cv'.")

    return {
        'fitness': fitness,
        'mean_roi': mean_roi,
        'std_roi': std_roi,
        'cv': cv,
        'per_split_roi': per_split_roi,
        'per_split_details': per_split_details,
    }


def ga_search_time_split_roi(
    df: pd.DataFrame,
    odds_df: pd.DataFrame,
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    population_size: int = 30,
    generations: int = 30,
    lambda_penalty: float = 1.0,
    objective: str = "mean_std",
    seed: Optional[int] = None,
    verbose: bool = True,
    time_column: str = 'DATE',
    return_all_results: bool = False,
) -> Tuple[Dict[str, float], float, Optional[List[Dict]]]:
    """
    Run genetic algorithm optimization with time-split ROI consistency objective.

    This function optimizes ELO hyperparameters to maximize:
        fitness = mean_roi - lambda * std_roi

    where mean_roi and std_roi are computed across the provided time splits.

    Args:
        df: Full historical fight data
        odds_df: Odds data for ROI calculation
        splits: Time splits from create_time_splits()
        population_size: GA population size (default 30)
        generations: Number of GA generations (default 30)
        lambda_penalty: Penalty weight for std_roi (default 1.0)
        objective: "mean_std" or "cv" (default "mean_std")
        seed: Random seed for reproducibility
        verbose: Print progress (default True)
        time_column: Name of the date column
        return_all_results: If True, return list of per-generation results

    Returns:
        Tuple of:
        - best_params: Dict of best ELO hyperparameters
        - best_fitness: Best fitness score achieved
        - all_results: List of generation results (if return_all_results=True)

    Sign Convention:
        Fitness is MAXIMIZED by the GA. Higher fitness is better.
        For "mean_std": High mean ROI and low std ROI = high fitness
        For "cv": Low coefficient of variation = high fitness
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if verbose:
        print(f"Starting GA optimization with {len(splits)} time splits")
        print(f"Objective: {objective}, Lambda: {lambda_penalty}")
        print(f"Population: {population_size}, Generations: {generations}")
        print()

    all_results = []

    # Initialize population
    population = []
    for _ in range(population_size):
        params = random_params()
        result = compute_time_split_fitness(
            df, odds_df, params, splits, lambda_penalty, objective, time_column
        )
        population.append({
            'params': params,
            'fitness': result['fitness'],
            'extended': result,
        })

    # Find initial best
    best_ind = max(population, key=lambda x: x['fitness'])

    if verbose:
        ext = best_ind['extended']
        print(f"Initial best: fitness={best_ind['fitness']:.2f}, "
              f"mean_roi={ext['mean_roi']:.2f}%, std_roi={ext['std_roi']:.2f}%")

    # GA evolution loop
    for gen in range(generations):
        new_population = []

        # Elitism: keep best individual
        population.sort(key=lambda x: x['fitness'], reverse=True)
        elite = population[0].copy()
        new_population.append(elite)

        # Generate offspring
        while len(new_population) < population_size:
            parent1 = tournament_select(population, k_tour=3)
            parent2 = tournament_select(population, k_tour=3)

            child_params = crossover(parent1, parent2)
            child_params = mutate(child_params)

            result = compute_time_split_fitness(
                df, odds_df, child_params, splits, lambda_penalty, objective, time_column
            )

            new_population.append({
                'params': child_params,
                'fitness': result['fitness'],
                'extended': result,
            })

        population = new_population
        gen_best = max(population, key=lambda x: x['fitness'])

        # Update global best
        is_new_best = gen_best['fitness'] > best_ind['fitness']
        if is_new_best:
            best_ind = gen_best.copy()

        # Calculate stats
        fitnesses = [ind['fitness'] for ind in population]
        avg_fitness = np.mean(fitnesses)

        if verbose:
            ext = gen_best['extended']
            new_best_str = " *** NEW BEST" if is_new_best else ""
            print(f"Gen {gen+1:02d}: fitness={gen_best['fitness']:.2f}, "
                  f"mean_roi={ext['mean_roi']:.2f}%, std_roi={ext['std_roi']:.2f}%, "
                  f"avg_fitness={avg_fitness:.2f}{new_best_str}")

        if return_all_results:
            ext = gen_best['extended']
            all_results.append({
                'generation': gen + 1,
                'best_fitness': gen_best['fitness'],
                'best_params': gen_best['params'].copy(),
                'mean_roi': ext['mean_roi'],
                'std_roi': ext['std_roi'],
                'cv': ext['cv'],
                'per_split_roi': ext['per_split_roi'],
                'avg_fitness': avg_fitness,
            })

    if return_all_results:
        return best_ind['params'], best_ind['fitness'], all_results
    return best_ind['params'], best_ind['fitness'], None


def save_results(
    output_dir: str,
    split_months: int,
    best_params: Dict[str, float],
    best_fitness: float,
    extended_result: Dict[str, Any],
    all_results: Optional[List[Dict]],
    config: Dict[str, Any]
) -> None:
    """
    Save optimization results to output directory.

    Creates:
    - {split_months}m_best_params.json: Best parameters found
    - {split_months}m_per_split_roi.csv: ROI for each time split
    - {split_months}m_evolution.csv: Per-generation metrics (if all_results provided)

    Args:
        output_dir: Output directory path
        split_months: Split duration in months
        best_params: Best parameters dict
        best_fitness: Best fitness value
        extended_result: Extended result dict with per_split details
        all_results: Per-generation results (optional)
        config: Configuration dict to include in output
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{split_months}m"

    # Save best parameters as JSON
    params_file = os.path.join(output_dir, f"{prefix}_best_params.json")
    output = {
        'config': config,
        'best_params': best_params,
        'best_fitness': best_fitness,
        'mean_roi': extended_result['mean_roi'],
        'std_roi': extended_result['std_roi'],
        'cv': extended_result['cv'],
        'num_splits': len(extended_result['per_split_roi']),
        'timestamp': datetime.now().isoformat(),
    }
    with open(params_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved best parameters to: {params_file}")

    # Save per-split ROI as CSV
    roi_file = os.path.join(output_dir, f"{prefix}_per_split_roi.csv")
    rows = []
    for i, detail in enumerate(extended_result['per_split_details']):
        rows.append({
            'split_idx': i,
            'val_start': detail['val_start'],
            'val_end': detail['val_end'],
            'roi_percent': detail['roi_percent'],
            'num_bets': detail['num_bets'],
            'wins': detail['wins'],
            'total_wagered': detail['total_wagered'],
            'total_profit': detail['total_profit'],
        })
    pd.DataFrame(rows).to_csv(roi_file, index=False)
    print(f"Saved per-split ROI to: {roi_file}")

    # Save evolution history as CSV
    if all_results:
        evolution_file = os.path.join(output_dir, f"{prefix}_evolution.csv")
        evo_rows = []
        for res in all_results:
            evo_rows.append({
                'generation': res['generation'],
                'best_fitness': res['best_fitness'],
                'mean_roi': res['mean_roi'],
                'std_roi': res['std_roi'],
                'cv': res['cv'],
                'avg_fitness': res['avg_fitness'],
            })
        pd.DataFrame(evo_rows).to_csv(evolution_file, index=False)
        print(f"Saved evolution history to: {evolution_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Time-split ROI optimization for GA-optimized ELO ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with 6-month splits
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 6

    # Run with multiple split lengths
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 2,6,12

    # Use coefficient of variation objective
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 6 --objective cv

    # Custom lambda penalty
    python ga_time_split_roi.py --data-file data/interleaved_cleaned.csv --split-months 6 --lambda 2.0

Fitness Function:
    The default objective maximizes: mean_roi - lambda * std_roi
    where mean_roi and std_roi are computed across time splits.
    This encourages the GA to find parameters with consistent ROI across time.
        """
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to historical fight data CSV"
    )
    parser.add_argument(
        "--odds-file",
        type=str,
        default=None,
        help="Path to odds data CSV. If not specified, uses after_averaging.csv in repo root"
    )
    parser.add_argument(
        "--time-column",
        type=str,
        default=None,
        help="Name of date/time column. Auto-detects 'DATE', 'date', or 'timestamp' if not specified"
    )
    parser.add_argument(
        "--split-months",
        type=str,
        required=True,
        help="Split duration in months. Can be single value (e.g., 6) or comma-separated list (e.g., 2,6,12)"
    )
    parser.add_argument(
        "--window-type",
        type=str,
        choices=["expanding", "rolling"],
        default="expanding",
        help="Window type for training data: 'expanding' (all prior data) or 'rolling' (fixed window)"
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=1.0,
        dest="lambda_penalty",
        help="Penalty weight for std_roi in fitness function (default: 1.0)"
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["mean_std", "cv"],
        default="mean_std",
        help="Fitness objective: 'mean_std' (mean_roi - lambda*std_roi) or 'cv' (minimize coefficient of variation)"
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=30,
        help="Number of GA generations (default: 30)"
    )
    parser.add_argument(
        "--population",
        type=int,
        default=30,
        help="GA population size (default: 30)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/time_split_roi",
        help="Output directory for results (default: results/time_split_roi)"
    )
    parser.add_argument(
        "--metric-output",
        type=str,
        default=None,
        help="Path to write consolidated metrics CSV"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    # New filtering arguments
    parser.add_argument(
        "--min-val-size",
        type=int,
        default=5,
        help="Minimum number of rows in validation set to keep a split (default: 5)"
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        help="Maximum number of splits to use after filtering (default: None, use all)"
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        choices=["even", "random"],
        default="even",
        help="Strategy for sampling splits: 'even' (evenly spaced) or 'random' (default: even)"
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for split sampling when strategy is 'random' (default: 42)"
    )

    args = parser.parse_args()

    # Parse split months
    split_months_list = [int(x.strip()) for x in args.split_months.split(',')]

    # Load data
    print(f"Loading data from: {args.data_file}")
    df = pd.read_csv(args.data_file, low_memory=False)

    # Detect and normalize time column
    time_column = args.time_column or detect_time_column(df)
    if time_column is None:
        raise ValueError("Could not auto-detect time column. Use --time-column to specify.")

    print(f"Using time column: {time_column}")

    # Ensure datetime format
    df[time_column] = pd.to_datetime(df[time_column]).dt.tz_localize(None)
    df = df.sort_values(time_column).reset_index(drop=True)

    # Ensure result column is numeric
    df['result'] = pd.to_numeric(df['result'], errors='coerce')

    # Add bout counts if not present
    df = add_bout_counts(df)

    # Load odds data
    if args.odds_file:
        odds_file = args.odds_file
    else:
        # Try to find after_averaging.csv in repo root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        odds_file = os.path.join(repo_root, "after_averaging.csv")

    if not os.path.exists(odds_file):
        raise FileNotFoundError(f"Odds file not found: {odds_file}")

    print(f"Loading odds from: {odds_file}")
    odds_df = pd.read_csv(odds_file, low_memory=False)
    odds_df["DATE"] = pd.to_datetime(odds_df["DATE"]).dt.tz_localize(None)

    print(f"\nData loaded:")
    print(f"  Fight data: {len(df)} rows, date range: {df[time_column].min()} to {df[time_column].max()}")
    print(f"  Odds data: {len(odds_df)} rows")

    # Store all results for consolidated output
    all_split_results = []

    # Run optimization for each split duration
    for split_months in split_months_list:
        print(f"\n{'='*60}")
        print(f"RUNNING OPTIMIZATION WITH {split_months}-MONTH SPLITS")
        print(f"{'='*60}\n")

        # Create time splits
        splits = create_time_splits(
            df,
            months=split_months,
            time_column=time_column,
            window_type=args.window_type
        )

        if len(splits) == 0:
            print(f"Warning: No valid splits created for {split_months}-month duration. Skipping.")
            continue

        # Log count before filtering
        splits_before_filter = len(splits)
        logger.info(f"Created {splits_before_filter} time splits before filtering")

        # Apply filtering and sampling
        splits = filter_and_sample_splits(
            splits,
            min_val_size=args.min_val_size,
            max_splits=args.max_splits,
            sample_strategy=args.sample_strategy,
            sample_seed=args.sample_seed,
            time_column=time_column
        )

        # Log count after filtering
        splits_after_filter = len(splits)
        logger.info(f"After filtering (min_val_size={args.min_val_size}, max_splits={args.max_splits}): {splits_after_filter} splits remaining")

        if splits_after_filter == 0:
            logger.warning(f"No splits remaining after filtering for {split_months}-month duration. Skipping.")
            continue

        # Print split info
        split_info = get_split_info(splits, time_column=time_column)
        print(f"Using {len(splits)} time splits (filtered from {splits_before_filter}):")
        for _, row in split_info.iterrows():
            print(f"  Split {int(row['split_idx'])}: "
                  f"train={row['train_rows']} rows ({row['train_start'].date()} to {row['train_end'].date()}), "
                  f"val={row['val_rows']} rows ({row['val_start'].date()} to {row['val_end'].date()})")
        print()

        # Run GA optimization
        best_params, best_fitness, all_results = ga_search_time_split_roi(
            df=df,
            odds_df=odds_df,
            splits=splits,
            population_size=args.population,
            generations=args.generations,
            lambda_penalty=args.lambda_penalty,
            objective=args.objective,
            seed=args.random_seed,
            verbose=not args.quiet,
            time_column=time_column,
            return_all_results=True,
        )

        # Compute final metrics with best params
        final_result = compute_time_split_fitness(
            df, odds_df, best_params, splits,
            args.lambda_penalty, args.objective, time_column
        )

        # Print summary
        print(f"\n=== RESULTS FOR {split_months}-MONTH SPLITS ===")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Mean ROI: {final_result['mean_roi']:.2f}%")
        print(f"Std ROI: {final_result['std_roi']:.2f}%")
        print(f"CV: {final_result['cv']:.3f}")
        print(f"Best parameters: {best_params}")
        print(f"\nPer-split ROI values: {[f'{r:.2f}%' for r in final_result['per_split_roi']]}")

        # Save results
        config = {
            'split_months': split_months,
            'window_type': args.window_type,
            'lambda_penalty': args.lambda_penalty,
            'objective': args.objective,
            'generations': args.generations,
            'population': args.population,
            'random_seed': args.random_seed,
        }

        save_results(
            output_dir=args.out_dir,
            split_months=split_months,
            best_params=best_params,
            best_fitness=best_fitness,
            extended_result=final_result,
            all_results=all_results,
            config=config,
        )

        # Store for consolidated output
        all_split_results.append({
            'split_months': split_months,
            'best_fitness': best_fitness,
            'mean_roi': final_result['mean_roi'],
            'std_roi': final_result['std_roi'],
            'cv': final_result['cv'],
            'num_splits': len(splits),
            'k': best_params['k'],
            'w_ko': best_params['w_ko'],
            'w_sub': best_params['w_sub'],
            'w_udec': best_params['w_udec'],
            'w_sdec': best_params['w_sdec'],
            'w_mdec': best_params['w_mdec'],
        })

    # Save consolidated metrics if requested
    if args.metric_output and all_split_results:
        consolidated_df = pd.DataFrame(all_split_results)
        consolidated_df.to_csv(args.metric_output, index=False)
        print(f"\nSaved consolidated metrics to: {args.metric_output}")

    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {args.out_dir}/")


if __name__ == "__main__":
    main()
