#!/usr/bin/env python3
"""
Consistency-focused Genetic Algorithm for decay parameter optimization.
Optimizes for stable performance across validation time windows using only validation data.
Does NOT touch OOS data during optimization.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import json
from deap import base, creator, tools, algorithms
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo.elo_utils import method_of_victory_scale, apply_multiphase_decay, add_bout_counts

def american_odds_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if pd.isna(odds):
        return None
    try:
        odds = float(odds)
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    except (ValueError, TypeError):
        return None

def run_elo_with_params(df, k=170, base_elo=1500, denominator=400, use_mov=True,
                        quick_succession_days=60, quick_succession_bump=1.05, 
                        decay_days=365, decay_rate=0.001):
    """Run Elo system with specified parameters."""
    df = df.copy()
    # Ensure result is numeric
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df = df[df['result'].notna()].copy()
    
    ratings = {}
    last_fight_date = {}
    pre, post, opp_pre, opp_post, probs = [], [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], float(row["result"])
        current_date = row["DATE"]
        
        # Get base ratings
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)
        
        # Apply decay/improvement adjustments
        days_since_f1 = (current_date - last_fight_date[f1]).days if f1 in last_fight_date else None
        days_since_f2 = (current_date - last_fight_date[f2]).days if f2 in last_fight_date else None
        
        r1 = apply_multiphase_decay(r1, days_since_f1, quick_succession_days,
                                   quick_succession_bump, decay_days, decay_rate)
        r2 = apply_multiphase_decay(r2, days_since_f2, quick_succession_days,
                                   quick_succession_bump, decay_days, decay_rate)

        # logistic expectation with overflow protection
        rating_diff = (r2 - r1) / denominator
        rating_diff = max(min(rating_diff, 100), -100)
        e1 = 1 / (1 + 10 ** rating_diff)
        e2 = 1 - e1

        # method of victory multiplier
        if use_mov:
            mov_scale = method_of_victory_scale(row)
            k_eff = k * mov_scale
        else:
            k_eff = k

        # update ratings
        r1_new = r1 + k_eff * (res - e1)
        r2_new = r2 + k_eff * ((1 - res) - e2)

        ratings[f1], ratings[f2] = r1_new, r2_new
        last_fight_date[f1] = current_date
        last_fight_date[f2] = current_date
        
        pre.append(r1)
        post.append(r1_new)
        opp_pre.append(r2)
        opp_post.append(r2_new)
        probs.append(e1)

    df["precomp_elo"] = pre
    df["postcomp_elo"] = post
    df["opp_precomp_elo"] = opp_pre
    df["opp_postcomp_elo"] = opp_post
    df["win_prob"] = probs
    
    return df, ratings, last_fight_date


def calculate_calibration_metrics(df):
    """Calculate Expected Calibration Error and Brier Score."""
    # Filter to fighters with precomp_boutcount > 1
    df_eval = df[(df['precomp_boutcount'] > 1) & (df['opp_precomp_boutcount'] > 1)].copy()
    
    if len(df_eval) == 0:
        return float('inf'), float('inf')
    
    probs = df_eval['win_prob'].values
    outcomes = df_eval['result'].values
    
    # Brier score
    brier = np.mean((probs - outcomes) ** 2)
    
    # Expected Calibration Error (ECE) - 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        bin_mask = (probs >= lower) & (probs < upper)
        
        if i == n_bins - 1:  # Last bin includes upper boundary
            bin_mask = (probs >= lower) & (probs <= upper)
        
        if bin_mask.sum() > 0:
            bin_prob_mean = probs[bin_mask].mean()
            bin_outcome_mean = outcomes[bin_mask].mean()
            bin_weight = bin_mask.sum() / len(probs)
            ece += bin_weight * abs(bin_prob_mean - bin_outcome_mean)
    
    return ece, brier


def calculate_window_roi(df_window, oos_df=None):
    """Calculate ROI for a time window."""
    # Filter to fighters with precomp_boutcount > 1
    df_eval = df_window[(df_window['precomp_boutcount'] > 1) & (df_window['opp_precomp_boutcount'] > 1)].copy()
    
    if len(df_eval) == 0:
        return 0.0, 0
    
    # For validation windows, use Elo-based implied odds
    # For OOS, use market odds if available
    if oos_df is not None and 'avg_odds' in oos_df.columns:
        # This is OOS evaluation
        df_eval = oos_df[(oos_df['precomp_boutcount'] > 1) & (oos_df['opp_precomp_boutcount'] > 1)].copy()
        df_eval = df_eval[df_eval['avg_odds'].notna()].copy()
        
        if len(df_eval) == 0:
            return 0.0, 0
        
        df_eval['decimal_odds'] = df_eval['avg_odds'].apply(american_odds_to_decimal)
        df_eval = df_eval[df_eval['decimal_odds'].notna()].copy()
        
        # Calculate ROI using market odds
        total_bet = len(df_eval)
        winnings = (df_eval['result'] * df_eval['decimal_odds']).sum()
        roi = ((winnings - total_bet) / total_bet) * 100
        
        return roi, len(df_eval)
    else:
        # Validation - use Elo implied odds
        df_eval['implied_odds'] = 1 / df_eval['win_prob']
        
        total_bet = len(df_eval)
        winnings = (df_eval['result'] * df_eval['implied_odds']).sum()
        roi = ((winnings - total_bet) / total_bet) * 100
        
        return roi, len(df_eval)


def evaluate_consistency_fitness(individual):
    """
    Evaluate parameters based on consistency across validation time windows.
    Uses ONLY validation data - OOS is never touched.
    
    Fitness components:
    1. Mean ROI across all validation windows
    2. Sharpe ratio (mean/std of window ROIs)
    3. Calibration quality (low ECE and Brier)
    4. Trend stability (penalize high volatility)
    """
    quick_succession_days, quick_succession_bump, decay_days, decay_rate = individual
    
    try:
        # Load and prepare training data
        df_train = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
        df_train['DATE'] = pd.to_datetime(df_train['DATE'])
        df_train = df_train.sort_values('DATE').reset_index(drop=True)
        
        # Add bout counts if not already present
        df_train = add_bout_counts(df_train)
        
        # Ensure bout count columns are numeric
        df_train['precomp_boutcount'] = pd.to_numeric(df_train['precomp_boutcount'], errors='coerce').fillna(0).astype(int)
        df_train['opp_precomp_boutcount'] = pd.to_numeric(df_train['opp_precomp_boutcount'], errors='coerce').fillna(0).astype(int)
        
        # Run Elo on full training data
        df_with_elo, ratings, last_fight = run_elo_with_params(
            df_train, k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=quick_succession_days,
            quick_succession_bump=quick_succession_bump,
            decay_days=decay_days,
            decay_rate=decay_rate
        )
        
        # Extract validation period (last year)
        max_date = df_with_elo['DATE'].max()
        one_year_ago = max_date - timedelta(days=365)
        val_df = df_with_elo[df_with_elo['DATE'] > one_year_ago].copy()
        
        if len(val_df) < 50:  # Need minimum data
            return (-1000.0,)
        
        # Split validation into 4 quarters for rolling window analysis
        val_df['quarter'] = pd.qcut(val_df['DATE'], q=4, labels=False, duplicates='drop')
        
        window_rois = []
        window_calibrations = []
        
        for quarter in range(4):
            window_df = val_df[val_df['quarter'] == quarter].copy()
            
            if len(window_df) < 10:  # Skip tiny windows
                continue
            
            # Calculate ROI for this window
            roi, n_fights = calculate_window_roi(window_df)
            if n_fights > 0:
                window_rois.append(roi)
            
            # Calculate calibration for this window
            ece, brier = calculate_calibration_metrics(window_df)
            if not np.isinf(ece):
                window_calibrations.append((ece, brier))
        
        if len(window_rois) < 2:  # Need at least 2 windows
            return (-1000.0,)
        
        # Fitness components
        mean_roi = np.mean(window_rois)
        std_roi = np.std(window_rois)
        
        # Sharpe ratio (reward per unit risk)
        # Add epsilon to prevent division by near-zero, which causes extremely high fitness values
        if std_roi < 0.01 or not np.isfinite(std_roi):  # Avoid division by zero or very small values
            sharpe = mean_roi if mean_roi > 0 else -1000
        else:
            sharpe = mean_roi / std_roi
        
        # Calibration quality (lower is better, so negate)
        if len(window_calibrations) > 0:
            mean_ece = np.mean([c[0] for c in window_calibrations])
            mean_brier = np.mean([c[1] for c in window_calibrations])
            calibration_penalty = (mean_ece * 100 + mean_brier * 50)  # Scale to ROI units
        else:
            calibration_penalty = 50
        
        # Trend consistency - penalize if ROI is decreasing over time
        if len(window_rois) >= 3:
            # Linear regression slope
            x = np.arange(len(window_rois))
            slope = np.polyfit(x, window_rois, 1)[0]
            trend_bonus = max(slope * 2, -10)  # Bonus for upward trends, cap penalty
        else:
            trend_bonus = 0
        
        # Combined fitness: prioritize Sharpe ratio and mean ROI, with calibration bonus
        # Sharpe ratio ensures consistency, mean ROI ensures profitability
        fitness = (
            0.4 * sharpe +  # Consistency (reward/risk)
            0.3 * mean_roi +  # Absolute performance
            0.2 * trend_bonus +  # Trend stability
            0.1 * (-calibration_penalty)  # Calibration quality
        )
        
        # Sanity check - ensure fitness is finite
        if not np.isfinite(fitness):
            return (-1000.0,)
        
        return (fitness,)
        
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return (-1000.0,)


def main():
    """Run consistency-focused GA optimization."""
    print("=" * 80)
    print("CONSISTENCY-FOCUSED DECAY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\nObjective: Find parameters with stable, well-calibrated performance")
    print("Method: Optimize Sharpe ratio, calibration, and trend stability on validation")
    print("OOS data: Completely untouched - used only for final evaluation\n")
    
    # Set up DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Parameter ranges
    toolbox.register("quick_days", random.uniform, 20, 120)
    toolbox.register("quick_bump", random.uniform, 1.01, 1.15)
    toolbox.register("decay_days", random.uniform, 180, 720)
    toolbox.register("decay_rate", random.uniform, 0.0001, 0.01)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.quick_days, toolbox.quick_bump, 
                     toolbox.decay_days, toolbox.decay_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_consistency_fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # GA parameters
    POPULATION_SIZE = 20
    GENERATIONS = 15
    CXPB = 0.7
    MUTPB = 0.3
    
    print(f"GA Configuration:")
    print(f"  Population size: {POPULATION_SIZE}")
    print(f"  Generations: {GENERATIONS}")
    print(f"  Crossover probability: {CXPB}")
    print(f"  Mutation probability: {MUTPB}\n")
    
    print("Parameter ranges:")
    print(f"  Quick succession days: 20-120")
    print(f"  Quick succession bump: 1.01-1.15")
    print(f"  Decay days: 180-720")
    print(f"  Decay rate: 0.0001-0.01\n")
    
    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print(f"Initial best fitness: {max([ind.fitness.values[0] for ind in pop]):.2f}\n")
    
    # Track best individual across generations
    best_individuals = []
    
    # Run GA
    for gen in range(GENERATIONS):
        print(f"\n{'='*80}")
        print(f"GENERATION {gen + 1}/{GENERATIONS}")
        print(f"{'='*80}")
        
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                # Enforce bounds after crossover
                child1[0] = max(20, min(120, child1[0]))
                child1[1] = max(1.01, min(1.15, child1[1]))
                child1[2] = max(180, min(720, child1[2]))
                child1[3] = max(0.0001, min(0.01, child1[3]))
                child2[0] = max(20, min(120, child2[0]))
                child2[1] = max(1.01, min(1.15, child2[1]))
                child2[2] = max(180, min(720, child2[2]))
                child2[3] = max(0.0001, min(0.01, child2[3]))
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                mutant[0] = max(20, min(120, mutant[0]))
                mutant[1] = max(1.01, min(1.15, mutant[1]))
                mutant[2] = max(180, min(720, mutant[2]))
                mutant[3] = max(0.0001, min(0.01, mutant[3]))
                del mutant.fitness.values
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
        
        # Get best individual of this generation
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        
        print(f"\nBest fitness: {best_fitness:.4f}")
        print(f"Best parameters:")
        print(f"  Quick succession days: {best_ind[0]:.1f}")
        print(f"  Quick succession bump: {best_ind[1]:.4f}x")
        print(f"  Decay days: {best_ind[2]:.1f}")
        print(f"  Decay rate: {best_ind[3]:.6f}")
        
        best_individuals.append({
            'generation': gen + 1,
            'fitness': best_fitness,
            'quick_succession_days': best_ind[0],
            'quick_succession_bump': best_ind[1],
            'decay_days': best_ind[2],
            'decay_rate': best_ind[3]
        })
    
    # Final evaluation with best parameters
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}\n")
    
    best_ever = max(best_individuals, key=lambda x: x['fitness'])
    print(f"Best parameters found:")
    print(f"  Quick succession days: {best_ever['quick_succession_days']:.1f}")
    print(f"  Quick succession bump: {best_ever['quick_succession_bump']:.4f}x")
    print(f"  Decay days: {best_ever['decay_days']:.1f}")
    print(f"  Decay rate: {best_ever['decay_rate']:.6f}")
    print(f"  Fitness score: {best_ever['fitness']:.4f}\n")
    
    # Now evaluate on validation and OOS for reporting (not for fitness)
    print("Evaluating on validation and OOS (for reporting only)...")
    
    df_train = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df_train['DATE'] = pd.to_datetime(df_train['DATE'])
    df_train = df_train.sort_values('DATE').reset_index(drop=True)
    
    df_with_elo, ratings, last_fight = run_elo_with_params(
        df_train, k=170, base_elo=1500, denominator=400, use_mov=True,
        quick_succession_days=best_ever['quick_succession_days'],
        quick_succession_bump=best_ever['quick_succession_bump'],
        decay_days=best_ever['decay_days'],
        decay_rate=best_ever['decay_rate']
    )
    
    # Validation metrics
    max_date = df_with_elo['DATE'].max()
    one_year_ago = max_date - timedelta(days=365)
    val_df = df_with_elo[df_with_elo['DATE'] > one_year_ago].copy()
    
    val_roi, val_n = calculate_window_roi(val_df)
    val_ece, val_brier = calculate_calibration_metrics(val_df)
    val_acc = ((val_df['result'] == (val_df['win_prob'] > 0.5).astype(int)).sum() / 
               len(val_df) * 100)
    
    print(f"\nValidation Performance:")
    print(f"  ROI: {val_roi:.2f}%")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  ECE: {val_ece:.4f}")
    print(f"  Brier Score: {val_brier:.4f}")
    print(f"  Fights evaluated: {val_n}")
    
    # OOS metrics (if available)
    try:
        df_oos = pd.read_csv('data/past3_events.csv', low_memory=False)
        df_oos['DATE'] = pd.to_datetime(df_oos['DATE'])
        df_oos = df_oos.sort_values('DATE').reset_index(drop=True)
        
        # Add bout counts if not already present
        df_oos = add_bout_counts(df_oos)
        
        # Ensure bout count columns are numeric
        df_oos['precomp_boutcount'] = pd.to_numeric(df_oos['precomp_boutcount'], errors='coerce').fillna(0).astype(int)
        df_oos['opp_precomp_boutcount'] = pd.to_numeric(df_oos['opp_precomp_boutcount'], errors='coerce').fillna(0).astype(int)
        
        # Merge with Elo ratings
        df_oos['precomp_elo'] = df_oos['FIGHTER'].map(ratings).fillna(1500)
        df_oos['opp_precomp_elo'] = df_oos['opp_FIGHTER'].map(ratings).fillna(1500)
        
        # Calculate win probabilities
        df_oos['rating_diff'] = (df_oos['opp_precomp_elo'] - df_oos['precomp_elo']) / 400
        df_oos['rating_diff'] = df_oos['rating_diff'].clip(-100, 100)
        df_oos['win_prob'] = 1 / (1 + 10 ** df_oos['rating_diff'])
        
        oos_roi, oos_n = calculate_window_roi(None, df_oos)
        
        df_oos_eval = df_oos[(df_oos['precomp_boutcount'] > 1) & (df_oos['opp_precomp_boutcount'] > 1)].copy()
        oos_acc = ((df_oos_eval['result'] == (df_oos_eval['win_prob'] > 0.5).astype(int)).sum() / 
                   len(df_oos_eval) * 100) if len(df_oos_eval) > 0 else 0
        
        print(f"\nOut-of-Sample Performance (BLIND TEST):")
        print(f"  ROI: {oos_roi:.2f}%")
        print(f"  Accuracy: {oos_acc:.2f}%")
        print(f"  Fights evaluated: {oos_n}")
        print(f"\n  Validation-OOS gap: {abs(val_roi - oos_roi):.2f}%")
        
    except Exception as e:
        print(f"\nCould not evaluate OOS: {e}")
        oos_roi = None
        oos_acc = None
        oos_n = 0
    
    # Save results
    results = {
        'optimization_type': 'consistency_focused_ga',
        'best_parameters': best_ever,
        'validation_performance': {
            'roi': float(val_roi),
            'accuracy': float(val_acc),
            'ece': float(val_ece),
            'brier': float(val_brier),
            'n_fights': int(val_n)
        },
        'oos_performance': {
            'roi': float(oos_roi) if oos_roi is not None else None,
            'accuracy': float(oos_acc) if oos_acc is not None else None,
            'n_fights': int(oos_n)
        },
        'generation_history': best_individuals
    }
    
    with open('decay_ga_consistency_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Results saved to: decay_ga_consistency_results.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
