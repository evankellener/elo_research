#!/usr/bin/env python3
"""
ROBUST Consistency-focused Genetic Algorithm for decay parameter optimization.
Optimizes for Sharpe ratio (mean/std) across rolling validation windows.
OOS data is NEVER used during optimization - only for final evaluation.

Key improvements:
- Strict parameter bounds enforcement
- Probability clipping to prevent numerical overflow  
- Proper nan/inf handling
- Better error messages and debugging
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

def clip_rating(rating, min_val=400, max_val=3000):
    """Clip rating to prevent numerical overflow."""
    return np.clip(rating, min_val, max_val)

def safe_expected_score(r1, r2, denominator=400):
    """Calculate expected score with overflow protection."""
    diff = clip_rating(r1) - clip_rating(r2)
    diff = np.clip(diff, -1000, 1000)  # Prevent extreme differences
    exp_val = 1.0 / (1.0 + 10 ** (-diff / denominator))
    return np.clip(exp_val, 0.001, 0.999)  # Prevent 0 or 1 probabilities

def run_elo_with_params(df, k=170, base_elo=1500, denominator=400, use_mov=True,
                        quick_succession_days=60, quick_succession_bump=1.05, 
                        decay_days=365, decay_rate=0.001):
    """Run Elo system with specified parameters."""
    # Parameter validation
    quick_succession_days = np.clip(quick_succession_days, 20, 120)
    quick_succession_bump = np.clip(quick_succession_bump, 1.01, 1.15)
    decay_days = np.clip(decay_days, 180, 720)
    decay_rate = np.clip(decay_rate, 0.0001, 0.01)  # Force positive
    
    df = df.copy()
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df = df[df['result'].notna()].copy()
    
    ratings = {}
    last_fight_date = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], float(row["result"])
        current_date = row["DATE"]
        
        r1 = clip_rating(ratings.get(f1, base_elo))
        r2 = clip_rating(ratings.get(f2, base_elo))
        
        # Apply decay
        if f1 in last_fight_date:
            days_since = (current_date - last_fight_date[f1]).days
            r1 = apply_multiphase_decay(
                r1, base_elo, days_since, 
                quick_succession_days, quick_succession_bump,
                decay_days, decay_rate
            )
            r1 = clip_rating(r1)
        
        if f2 in last_fight_date:
            days_since = (current_date - last_fight_date[f2]).days
            r2 = apply_multiphase_decay(
                r2, base_elo, days_since,
                quick_succession_days, quick_succession_bump,
                decay_days, decay_rate
            )
            r2 = clip_rating(r2)
        
        pre.append(r1)
        opp_pre.append(r2)
        
        # Calculate expected scores with safe function
        e1 = safe_expected_score(r1, r2, denominator)
        e2 = 1.0 - e1
        
        # Apply MOV scaling
        mov_scale = 1.0
        if use_mov and 'method' in row:
            mov_scale = method_of_victory_scale(row['method'])
        
        # Update ratings
        delta = k * mov_scale * (res - e1)
        r1_new = clip_rating(r1 + delta)
        r2_new = clip_rating(r2 - delta)
        
        post.append(r1_new)
        opp_post.append(r2_new)
        
        ratings[f1] = r1_new
        ratings[f2] = r2_new
        last_fight_date[f1] = current_date
        last_fight_date[f2] = current_date
    
    df['pre_elo'] = pre
    df['opp_pre_elo'] = opp_pre
    df['post_elo'] = post
    df['opp_post_elo'] = opp_post
    
    # Calculate win probabilities with safe function
    df['win_prob'] = df.apply(
        lambda x: safe_expected_score(x['pre_elo'], x['opp_pre_elo'], denominator), 
        axis=1
    )
    
    return df

def calculate_roi_for_window(df, market_odds_col='avg_odds'):
    """Calculate ROI for a time window."""
    df = df[df['boutcount'] > 1].copy()
    
    if market_odds_col not in df.columns or len(df) == 0:
        return None, 0
    
    df['decimal_odds'] = df[market_odds_col].apply(american_odds_to_decimal)
    df = df[df['decimal_odds'].notna()].copy()
    
    if len(df) == 0:
        return None, 0
    
    total_bet = len(df)
    total_return = sum(df['result'] * df['decimal_odds'])
    roi = ((total_return - total_bet) / total_bet) * 100 if total_bet > 0 else None
    
    # Clip extreme values
    if roi is not None:
        roi = np.clip(roi, -100, 500)
    
    return roi, len(df)

def evaluate_consistency(individual, val_df, num_windows=4):
    """
    Evaluate consistency using Sharpe ratio across rolling windows.
    ONLY uses validation data - OOS never touched.
    """
    days, bump, decay_days, rate = individual
    
    # Strict bounds enforcement
    if not (20 <= days <= 120):
        return (-1e10,)
    if not (1.01 <= bump <= 1.15):
        return (-1e10,)
    if not (180 <= decay_days <= 720):
        return (-1e10,)
    if not (0.0001 <= rate <= 0.01):
        return (-1e10,)
    
    try:
        # Run Elo on validation data
        val_result = run_elo_with_params(
            val_df, k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=days, quick_succession_bump=bump,
            decay_days=decay_days, decay_rate=rate
        )
        
        # Split into time windows
        val_result = val_result.sort_values('DATE')
        window_size = len(val_result) // num_windows
        
        rois = []
        for i in range(num_windows):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size if i < num_windows - 1 else len(val_result)
            window_df = val_result.iloc[start_idx:end_idx]
            
            roi, n_fights = calculate_roi_for_window(window_df)
            if roi is not None and n_fights >= 5:  # Minimum fights per window
                rois.append(roi)
        
        if len(rois) < 2:  # Need at least 2 windows for std
            return (-1e10,)
        
        mean_roi = np.mean(rois)
        std_roi = np.std(rois)
        
        # Sharpe ratio with safeguards
        if std_roi < 0.01 or np.isnan(std_roi) or np.isinf(std_roi):
            sharpe = mean_roi if mean_roi > 0 else -1e10
        else:
            sharpe = mean_roi / std_roi
        
        # Bonus for positive mean ROI
        fitness = sharpe + 0.1 * max(mean_roi, 0)
        
        # Handle inf/nan
        if np.isnan(fitness) or np.isinf(fitness):
            return (-1e10,)
        
        return (fitness,)
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return (-1e10,)

def main():
    print("=" * 80)
    print("ROBUST CONSISTENCY-FOCUSED DECAY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print()
    print("Objective: Maximize Sharpe ratio (mean ROI / std ROI) across validation windows")
    print("Method: Rolling window analysis on validation data only")
    print("OOS: Completely untouched - used only for final blind test")
    print()
    
    # Load data
    try:
        df = pd.read_csv('data/interleaved_cleaned.csv')
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Split validation (last year) vs training
    max_date = df['DATE'].max()
    val_start = max_date - timedelta(days=365)
    val_df = df[df['DATE'] > val_start].copy()
    
    print(f"Validation data: {len(val_df)} fights")
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
    
    toolbox.register("evaluate", evaluate_consistency, val_df=val_df, num_windows=4)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA
    pop_size = 20
    n_gen = 15
    
    print(f"GA Configuration: Pop={pop_size}, Gen={n_gen}, Cx=0.7, Mut=0.3")
    print()
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("mean", np.mean)
    
    print("Starting evolution...")
    print()
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, 
                                    ngen=n_gen, stats=stats, halloffame=hof, verbose=True)
    
    # Extract best parameters
    best = hof[0]
    days, bump, decay_days, rate = best
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print()
    print(f"Best parameters (by Sharpe ratio):")
    print(f"  Quick succession days: {days:.1f}")
    print(f"  Quick succession bump: {bump:.4f}x")
    print(f"  Decay days: {decay_days:.1f}")
    print(f"  Decay rate: {rate:.6f}")
    print(f"  Fitness (Sharpe): {best.fitness.values[0]:.4f}")
    print()
    
    # Final validation evaluation
    print("Calculating detailed validation metrics...")
    val_result = run_elo_with_params(
        val_df, k=170, base_elo=1500, denominator=400, use_mov=True,
        quick_succession_days=days, quick_succession_bump=bump,
        decay_days=decay_days, decay_rate=rate
    )
    
    val_roi, val_fights = calculate_roi_for_window(val_result)
    val_acc = val_result[val_result['boutcount'] > 1]['result'].mean() * 100 if len(val_result[val_result['boutcount'] > 1]) > 0 else None
    
    print(f"Validation Performance:")
    print(f"  ROI: {val_roi:.2f}%" if val_roi else "  ROI: N/A")
    print(f"  Accuracy: {val_acc:.2f}%" if val_acc else "  Accuracy: N/A")
    print(f"  Fights: {val_fights}")
    print()
    
    # Final OOS evaluation (BLIND TEST - never seen during optimization)
    try:
        oos_df = pd.read_csv('data/past3_events.csv')
        oos_df['DATE'] = pd.to_datetime(oos_df['DATE'])
        oos_df = oos_df.sort_values('DATE')
        
        print("Running blind OOS evaluation...")
        
        # Combine training + validation for final Elo calculation
        full_train_df = df.copy()
        oos_df_with_history = pd.concat([full_train_df, oos_df], ignore_index=True)
        oos_df_with_history = oos_df_with_history.sort_values('DATE')
        
        oos_result = run_elo_with_params(
            oos_df_with_history, k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=days, quick_succession_bump=bump,
            decay_days=decay_days, decay_rate=rate
        )
        
        # Extract only OOS portion
        oos_only = oos_result[oos_result['DATE'] >= oos_df['DATE'].min()].copy()
        
        oos_roi, oos_fights = calculate_roi_for_window(oos_only)
        oos_acc = oos_only[oos_only['boutcount'] > 1]['result'].mean() * 100 if len(oos_only[oos_only['boutcount'] > 1]) > 0 else None
        
        print(f"OOS Performance (BLIND TEST):")
        print(f"  ROI: {oos_roi:.2f}%" if oos_roi else "  ROI: N/A")
        print(f"  Accuracy: {oos_acc:.2f}%" if oos_acc else "  Accuracy: N/A")
        print(f"  Fights: {oos_fights}")
        print()
        
        if val_roi and oos_roi:
            gap = abs(val_roi - oos_roi)
            print(f"Validation-OOS Gap: {gap:.2f}%")
            print()
            
            if gap < 10:
                print("✓ EXCELLENT CONSISTENCY - Gap < 10%")
            elif gap < 25:
                print("✓ GOOD CONSISTENCY - Gap < 25%")
            else:
                print("⚠ MODERATE GAP - Consider additional validation")
        
    except Exception as e:
        print(f"Could not evaluate OOS: {e}")
        oos_roi, oos_acc, oos_fights = None, None, 0
    
    # Save results
    results = {
        'approach': 'consistency_sharpe_ratio',
        'best_parameters': {
            'quick_succession_days': float(days),
            'quick_succession_bump': float(bump),
            'decay_days': float(decay_days),
            'decay_rate': float(rate)
        },
        'fitness': float(best.fitness.values[0]) if not np.isinf(best.fitness.values[0]) else None,
        'validation_performance': {
            'roi': float(val_roi) if val_roi else None,
            'accuracy': float(val_acc) if val_acc else None,
            'n_fights': int(val_fights)
        },
        'oos_performance': {
            'roi': float(oos_roi) if oos_roi else None,
            'accuracy': float(oos_acc) if oos_acc else None,
            'n_fights': int(oos_fights)
        }
    }
    
    with open('decay_ga_consistency_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("Results saved to decay_ga_consistency_results.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
