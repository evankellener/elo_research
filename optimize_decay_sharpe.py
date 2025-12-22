#!/usr/bin/env python3
"""
Consistency-focused GA optimization using Sharpe ratio.
Optimizes for stable performance across rolling validation windows.
OOS is never touched during optimization - only used for final blind test.
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
    # Enforce parameter bounds strictly
    quick_succession_days = max(20, min(120, quick_succession_days))
    quick_succession_bump = max(1.01, min(1.15, quick_succession_bump))
    decay_days = max(180, min(720, decay_days))
    decay_rate = max(0.0001, min(0.01, abs(decay_rate)))  # Force positive
    
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
        if f1 in last_fight_date:
            days_since = (current_date - last_fight_date[f1]).days
            if days_since <= quick_succession_days:
                r1 *= quick_succession_bump
            elif days_since > decay_days:
                decay_factor = np.exp(-decay_rate * (days_since - decay_days))
                r1 *= decay_factor
                
        if f2 in last_fight_date:
            days_since = (current_date - last_fight_date[f2]).days
            if days_since <= quick_succession_days:
                r2 *= quick_succession_bump
            elif days_since > decay_days:
                decay_factor = np.exp(-decay_rate * (days_since - decay_days))
                r2 *= decay_factor
        
        # Clip ratings to reasonable bounds
        r1 = np.clip(r1, 100, 3000)
        r2 = np.clip(r2, 100, 3000)
        
        # Store pre-fight ratings
        pre.append(r1)
        opp_pre.append(r2)
        
        # Calculate expected and actual
        diff = (r2 - r1) / denominator
        exp_prob = 1 / (1 + 10 ** diff)
        exp_prob = np.clip(exp_prob, 0.01, 0.99)  # Clip probabilities
        
        actual = 1 if res == "W" else 0
        
        # Update ratings
        mov_scale = 1.0
        if use_mov and "mov" in row:
            mov_scale = method_of_victory_scale(row["mov"])
        
        delta = k * mov_scale * (actual - exp_prob)
        new_r1 = r1 + delta
        new_r2 = r2 - delta
        
        # Store post-fight ratings
        post.append(new_r1)
        opp_post.append(new_r2)
        
        # Update dictionaries
        ratings[f1] = new_r1
        ratings[f2] = new_r2
        last_fight_date[f1] = current_date
        last_fight_date[f2] = current_date
    
    df["pre_elo"] = pre
    df["post_elo"] = post
    df["opp_pre_elo"] = opp_pre
    df["opp_post_elo"] = opp_post
    
    return df

def calculate_roi(df_with_elo):
    """Calculate ROI using market odds."""
    roi_data = []
    for _, row in df_with_elo.iterrows():
        if pd.isna(row.get("avg_odds")):
            continue
        
        decimal_odds = american_odds_to_decimal(row["avg_odds"])
        if decimal_odds is None or decimal_odds <= 1:
            continue
        
        # Bet on fighter if our Elo thinks they'll win
        elo_pred = row["pre_elo"] > row["opp_pre_elo"]
        actual_win = row["result"] == "W"
        
        if elo_pred:
            profit = (decimal_odds - 1) if actual_win else -1
            roi_data.append(profit)
    
    if not roi_data:
        return 0, 0
    
    return np.mean(roi_data) * 100, len(roi_data)

def evaluate_sharpe_fitness(individual):
    """
    Fitness function that optimizes for Sharpe ratio across rolling validation windows.
    Returns (sharpe_ratio,) as a tuple for DEAP.
    """
    quick_days, bump, decay_days, rate = individual
    
    # Enforce strict bounds
    quick_days = max(20, min(120, quick_days))
    bump = max(1.01, min(1.15, bump))
    decay_days = max(180, min(720, decay_days))
    rate = max(0.0001, min(0.01, abs(rate)))
    
    try:
        # Load data
        df = pd.read_csv("data/interleaved_cleaned.csv", parse_dates=["DATE"])
        df = df.sort_values("DATE")
        
        # Get validation period (last year)
        max_date = df["DATE"].max()
        val_start = max_date - timedelta(days=365)
        
        train_df = df[df["DATE"] < val_start].copy()
        val_df = df[df["DATE"] >= val_start].copy()
        
        # Run Elo on training data to build ratings
        train_df = run_elo_with_params(train_df, quick_succession_days=quick_days,
                                       quick_succession_bump=bump, decay_days=decay_days,
                                       decay_rate=rate)
        
        # Split validation into 4 quarterly windows
        val_df = val_df.copy()
        val_df["quarter"] = pd.cut(val_df["DATE"], bins=4, labels=False)
        
        quarterly_rois = []
        for q in range(4):
            quarter_df = val_df[val_df["quarter"] == q].copy()
            if len(quarter_df) < 10:  # Need minimum fights
                continue
                
            quarter_df = run_elo_with_params(quarter_df, quick_succession_days=quick_days,
                                            quick_succession_bump=bump, decay_days=decay_days,
                                            decay_rate=rate)
            
            # Filter for boutcount > 1
            quarter_df = quarter_df[quarter_df["precomp_boutcount"] > 1]
            if len(quarter_df) < 5:
                continue
            
            roi, n_bets = calculate_roi(quarter_df)
            if n_bets >= 5:  # Minimum bets per window
                quarterly_rois.append(roi)
        
        if len(quarterly_rois) < 3:  # Need at least 3 quarters
            return (-1000000,)
        
        # Calculate Sharpe ratio
        mean_roi = np.mean(quarterly_rois)
        std_roi = np.std(quarterly_rois)
        
        if std_roi < 0.01 or not np.isfinite(std_roi):  # Avoid division by zero
            sharpe = mean_roi if mean_roi > 0 else -1000
        else:
            sharpe = mean_roi / std_roi
        
        # Add small bonus for positive mean ROI
        if mean_roi > 0:
            sharpe += 0.1 * mean_roi
        
        if not np.isfinite(sharpe):
            return (-1000000,)
        
        return (sharpe,)
    
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return (-1000000,)

def main():
    print("="*80)
    print("CONSISTENCY-FOCUSED DECAY OPTIMIZATION (Sharpe Ratio)")
    print("="*80)
    print("\nObjective: Maximize Sharpe ratio across quarterly validation windows")
    print("Method: Split validation year into 4 quarters, optimize for consistency")
    print("OOS: Completely untouched - final blind test only\n")
    
    # Set up GA
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_quick_days", random.uniform, 20, 120)
    toolbox.register("attr_bump", random.uniform, 1.01, 1.15)
    toolbox.register("attr_decay_days", random.uniform, 180, 720)
    toolbox.register("attr_rate", random.uniform, 0.0001, 0.01)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.attr_quick_days, toolbox.attr_bump, 
                     toolbox.attr_decay_days, toolbox.attr_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_sharpe_fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Run GA
    pop = toolbox.population(n=20)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    print("Running GA optimization (15 generations, 20 individuals)...")
    print("This will take approximately 5-7 minutes...\n")
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=15,
                                   stats=stats, halloffame=hof, verbose=True)
    
    # Get best individual
    best = hof[0]
    quick_days, bump, decay_days, rate = best
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nBest Parameters (Optimized for Sharpe Ratio):")
    print(f"  Quick succession days: {quick_days:.1f}")
    print(f"  Quick succession bump: {bump:.4f}x ({(bump-1)*100:.2f}% boost)")
    print(f"  Decay days: {decay_days:.1f} (~{decay_days/30:.1f} months)")
    print(f"  Decay rate: {rate:.6f}")
    print(f"  Best Sharpe ratio: {best.fitness.values[0]:.4f}")
    
    # Final evaluation on full validation and OOS
    print("\n" + "-"*80)
    print("FINAL VALIDATION & OOS EVALUATION")
    print("-"*80)
    
    df = pd.read_csv("data/interleaved_cleaned.csv", parse_dates=["DATE"])
    df = df.sort_values("DATE")
    
    max_date = df["DATE"].max()
    val_start = max_date - timedelta(days=365)
    
    val_df = df[df["DATE"] >= val_start].copy()
    val_df = run_elo_with_params(val_df, quick_succession_days=quick_days,
                                 quick_succession_bump=bump, decay_days=decay_days,
                                 decay_rate=rate)
    val_df = val_df[val_df["precomp_boutcount"] > 1]
    val_roi, val_n = calculate_roi(val_df)
    val_acc = (val_df["result"] == "W").mean() * 100
    
    print(f"\nValidation (last year, boutcount>1):")
    print(f"  ROI: {val_roi:.2f}%")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  Fights: {val_n}")
    
    # OOS evaluation
    oos_df = pd.read_csv("data/past3_events.csv", parse_dates=["DATE"])
    oos_df = oos_df[~oos_df["avg_odds"].isna()].copy()
    oos_df = run_elo_with_params(oos_df, quick_succession_days=quick_days,
                                 quick_succession_bump=bump, decay_days=decay_days,
                                 decay_rate=rate)
    oos_df = oos_df[oos_df["precomp_boutcount"] > 1]
    oos_roi, oos_n = calculate_roi(oos_df)
    oos_acc = (oos_df["result"] == "W").mean() * 100
    
    print(f"\nOut-of-Sample (real market odds, boutcount>1):")
    print(f"  ROI: {oos_roi:.2f}%")
    print(f"  Accuracy: {oos_acc:.2f}%")
    print(f"  Fights: {oos_n}")
    
    print(f"\nValidation-OOS Gap: {abs(val_roi - oos_roi):.2f}%")
    
    # Save results
    results = {
        'optimization_type': 'sharpe_ratio_consistency',
        'best_parameters': {
            'quick_succession_days': float(quick_days),
            'quick_succession_bump': float(bump),
            'decay_days': float(decay_days),
            'decay_rate': float(rate),
            'sharpe_ratio': float(best.fitness.values[0])
        },
        'validation': {
            'roi': float(val_roi),
            'accuracy': float(val_acc),
            'n_fights': int(val_n)
        },
        'oos': {
            'roi': float(oos_roi),
            'accuracy': float(oos_acc),
            'n_fights': int(oos_n)
        },
        'gap': float(abs(val_roi - oos_roi))
    }
    
    with open('decay_sharpe_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to decay_sharpe_optimization_results.json")
    print("="*80)

if __name__ == "__main__":
    main()
