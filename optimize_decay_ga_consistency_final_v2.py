#!/usr/bin/env python3
"""
CONSISTENCY-FOCUSED Genetic Algorithm for decay parameter optimization.
Optimizes for Sharpe ratio (mean/std) across rolling VALIDATION windows only.
OOS data is NEVER used during optimization - only for final blind evaluation.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import json
from deap import base, creator, tools, algorithms
import random
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from elo.elo_utils import method_of_victory_scale, apply_multiphase_decay

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
    ratings = {}
    last_fight_date = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
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

    df["precomp_elo"] = pre
    df["postcomp_elo"] = post
    df["opp_precomp_elo"] = opp_pre
    df["opp_postcomp_elo"] = opp_post
    
    return df, ratings, last_fight_date

def calculate_roi_for_window(df_window, market_odds_col='avg_odds'):
    """Calculate ROI for a dataframe window."""
    # Filter for fighters with boutcount > 1
    df_eval = df_window[df_window['boutcount'] > 1].copy()
    
    if len(df_eval) == 0 or market_odds_col not in df_eval.columns:
        return None, 0
    
    # Convert American odds to decimal
    df_eval['decimal_odds'] = df_eval[market_odds_col].apply(american_odds_to_decimal)
    df_eval = df_eval[df_eval['decimal_odds'].notna()].copy()
    
    if len(df_eval) == 0:
        return None, 0
    
    total_bet = len(df_eval)
    total_return = sum(df_eval['result'] * df_eval['decimal_odds'])
    roi = ((total_return - total_bet) / total_bet) * 100
    
    return roi, len(df_eval)

def evaluate_consistency(individual, full_df):
    """
    Evaluate consistency using Sharpe ratio across rolling VALIDATION windows.
    NEVER touches OOS data - maintains experimental integrity.
    
    Returns: (fitness_score,) 
    """
    quick_days, quick_bump, decay_days, decay_rate = individual
    
    # Parameter bounds check
    if not (20 <= quick_days <= 120):
        return (-1e10,)
    if not (1.01 <= quick_bump <= 1.15):
        return (-1e10,)
    if not (180 <= decay_days <= 720):
        return (-1e10,)
    if not (0.0001 <= decay_rate <= 0.01):
        return (-1e10,)
    
    try:
        # Split data: use last year as validation
        last_date = full_df['DATE'].max()
        one_year_ago = last_date - timedelta(days=365)
        
        train_df = full_df[full_df['DATE'] < one_year_ago].copy()
        val_df = full_df[full_df['DATE'] >= one_year_ago].copy()
        
        if len(val_df) < 40:  # Need minimum data
            return (-1e10,)
        
        # Run Elo on training data first
        train_result, ratings, last_fight_dates = run_elo_with_params(
            train_df,
            k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=int(quick_days),
            quick_succession_bump=quick_bump,
            decay_days=int(decay_days),
            decay_rate=decay_rate
        )
        
        # Continue Elo on validation data
        val_result, _, _ = run_elo_with_params(
            val_df,
            k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=int(quick_days),
            quick_succession_bump=quick_bump,
            decay_days=int(decay_days),
            decay_rate=decay_rate
        )
        
        # Split validation into rolling windows (quarterly)
        val_result = val_result.sort_values('DATE')
        num_windows = 4
        window_size = len(val_result) // num_windows
        
        rois = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < num_windows - 1 else len(val_result)
            window_df = val_result.iloc[start_idx:end_idx]
            
            roi, n_fights = calculate_roi_for_window(window_df)
            if roi is not None and n_fights >= 5:  # Minimum 5 fights per window
                rois.append(roi)
        
        if len(rois) < 2:  # Need at least 2 windows for std
            return (-1e10,)
        
        # Calculate Sharpe ratio: mean ROI / std ROI
        mean_roi = np.mean(rois)
        std_roi = np.std(rois)
        
        if std_roi < 0.01:  # Avoid division by near-zero
            sharpe = mean_roi if mean_roi > 0 else -1e10
        else:
            sharpe = mean_roi / std_roi
        
        # Fitness = Sharpe + small bonus for positive mean
        fitness = sharpe + 0.1 * max(mean_roi, 0)
        
        # Sanity checks
        if np.isnan(fitness) or np.isinf(fitness):
            return (-1e10,)
        
        return (fitness,)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return (-1e10,)

def main():
    print("=" * 80)
    print("CONSISTENCY-FOCUSED DECAY PARAMETER OPTIMIZATION (SHARPE RATIO)")
    print("=" * 80)
    print()
    print("Objective: Maximize Sharpe ratio (mean ROI / std ROI) across validation windows")
    print("Method: Rolling window analysis on VALIDATION data only")
    print("OOS: Completely untouched during optimization - used only for final blind test")
    print()
    
    # Load full training data
    try:
        df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df[df['result'].notna()].copy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Total training data: {len(df)} fights")
    last_date = df['DATE'].max()
    one_year_ago = last_date - timedelta(days=365)
    val_size = len(df[df['DATE'] >= one_year_ago])
    print(f"Validation data (last year): {val_size} fights")
    print()
    
    # Setup GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_days", random.uniform, 20, 120)
    toolbox.register("attr_bump", random.uniform, 1.01, 1.15)
    toolbox.register("attr_decay_days", random.uniform, 180, 720)
    toolbox.register("attr_rate", random.uniform, 0.0001, 0.01)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_days, toolbox.attr_bump, 
                     toolbox.attr_decay_days, toolbox.attr_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_consistency, full_df=df)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA
    pop_size = 20
    n_gen = 15
    
    print(f"GA Configuration: Pop={pop_size}, Gen={n_gen}, Cx=0.7, Mut=0.3")
    print("Optimizing for Sharpe ratio on validation windows...")
    print()
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                    ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
    
    # Extract best parameters
    best = hof[0]
    days, bump, decay_days, rate = best
    
    print()
    print("=" * 80)
    print("FINAL RESULTS - CONSISTENCY OPTIMIZATION")
    print("=" * 80)
    print()
    print(f"Best parameters (by Sharpe ratio on validation windows):")
    print(f"  Quick succession days: {days:.1f}")
    print(f"  Quick succession bump: {bump:.4f}x ({(bump-1)*100:.2f}% boost)")
    print(f"  Decay days: {decay_days:.1f} (~{decay_days/30:.1f} months)")
    print(f"  Decay rate: {rate:.6f}")
    print(f"  Sharpe-based fitness: {best.fitness.values[0]:.4f}")
    print()
    
    # Final detailed evaluation on validation
    print("Running final validation evaluation...")
    one_year_ago = df['DATE'].max() - timedelta(days=365)
    train_df = df[df['DATE'] < one_year_ago].copy()
    val_df = df[df['DATE'] >= one_year_ago].copy()
    
    train_result, ratings, last_fight_dates = run_elo_with_params(
        train_df, k=170, base_elo=1500, denominator=400, use_mov=True,
        quick_succession_days=int(days), quick_succession_bump=bump,
        decay_days=int(decay_days), decay_rate=rate
    )
    
    val_result, _, _ = run_elo_with_params(
        val_df, k=170, base_elo=1500, denominator=400, use_mov=True,
        quick_succession_days=int(days), quick_succession_bump=bump,
        decay_days=int(decay_days), decay_rate=rate
    )
    
    val_roi, val_fights = calculate_roi_for_window(val_result)
    val_acc = val_result[val_result['boutcount'] > 1]['result'].mean() * 100 if len(val_result[val_result['boutcount'] > 1]) > 0 else None
    
    print(f"\nValidation Performance (Full Year):")
    print(f"  ROI: {val_roi:.2f}%" if val_roi else "  ROI: N/A")
    print(f"  Accuracy: {val_acc:.2f}%" if val_acc else "  Accuracy: N/A")
    print(f"  Fights: {val_fights}")
    print()
    
    # Final OOS evaluation (BLIND TEST - never seen during optimization)
    try:
        oos_df = pd.read_csv('data/past3_events.csv')
        oos_df['DATE'] = pd.to_datetime(oos_df['date'])
        oos_df = oos_df.rename(columns={
            'fighter': 'FIGHTER',
            'opp_fighter': 'opp_FIGHTER'
        })
        oos_df['result'] = pd.to_numeric(oos_df['result'], errors='coerce')
        oos_df = oos_df.sort_values('DATE')
        
        print("Running BLIND out-of-sample test (OOS data never seen during optimization)...")
        
        # Combine all training data and run through OOS
        all_train = df.copy()
        combined = pd.concat([all_train, oos_df], ignore_index=True)
        combined = combined.sort_values('DATE')
        
        combined_result, _, _ = run_elo_with_params(
            combined, k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=int(days), quick_succession_bump=bump,
            decay_days=int(decay_days), decay_rate=rate
        )
        
        # Extract OOS portion only
        oos_min_date = oos_df['DATE'].min()
        oos_only = combined_result[combined_result['DATE'] >= oos_min_date].copy()
        
        oos_roi, oos_fights = calculate_roi_for_window(oos_only)
        oos_acc = oos_only[oos_only['boutcount'] > 1]['result'].mean() * 100 if len(oos_only[oos_only['boutcount'] > 1]) > 0 else None
        
        print(f"\nOOS Performance (BLIND TEST - Never Used in Optimization):")
        print(f"  ROI: {oos_roi:.2f}%" if oos_roi else "  ROI: N/A")
        print(f"  Accuracy: {oos_acc:.2f}%" if oos_acc else "  Accuracy: N/A")
        print(f"  Fights: {oos_fights}")
        print()
        
        if val_roi and oos_roi:
            gap = abs(val_roi - oos_roi)
            print(f"Validation-OOS Gap: {gap:.2f}%")
            print()
            
            if gap < 10:
                print("✓✓ EXCELLENT CONSISTENCY - Gap < 10%")
                print("   Parameters show strong generalization!")
            elif gap < 25:
                print("✓ GOOD CONSISTENCY - Gap < 25%")
                print("   Parameters show reasonable generalization.")
            else:
                print("⚠ MODERATE GAP - Consider additional validation")
                print("  Gap may indicate distribution shift between periods.")
        
    except Exception as e:
        print(f"Could not evaluate OOS: {e}")
        oos_roi, oos_acc, oos_fights = None, None, 0
    
    # Save results
    results = {
        'approach': 'consistency_sharpe_ratio',
        'optimization_target': 'sharpe_ratio_on_validation_windows',
        'oos_never_used_during_optimization': True,
        'best_parameters': {
            'quick_succession_days': float(days),
            'quick_succession_bump': float(bump),
            'decay_days': float(decay_days),
            'decay_rate': float(rate)
        },
        'sharpe_fitness': float(best.fitness.values[0]) if not np.isinf(best.fitness.values[0]) else None,
        'validation_performance': {
            'roi': float(val_roi) if val_roi else None,
            'accuracy': float(val_acc) if val_acc else None,
            'n_fights': int(val_fights)
        },
        'oos_performance': {
            'roi': float(oos_roi) if oos_roi else None,
            'accuracy': float(oos_acc) if oos_acc else None,
            'n_fights': int(oos_fights),
            'note': 'Blind test - never used during optimization'
        },
        'val_oos_gap': float(abs(val_roi - oos_roi)) if (val_roi and oos_roi) else None
    }
    
    with open('decay_ga_consistency_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 80)
    print("Results saved to decay_ga_consistency_results_final.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
