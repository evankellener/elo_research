#!/usr/bin/env python3
"""
Consistency-focused Genetic Algorithm for decay parameter optimization.
Optimizes for Sharpe ratio (mean/std) across rolling validation windows.
OOS data is NEVER used during optimization - only for final evaluation.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
import json
from deap import base, creator, tools
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
    df = df.copy()
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df = df[df['result'].notna()].copy()
    
    ratings = {}
    last_fight_date = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], float(row["result"])
        current_date = row["DATE"]
        
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)
        
        days_since_f1 = (current_date - last_fight_date[f1]).days if f1 in last_fight_date else None
        days_since_f2 = (current_date - last_fight_date[f2]).days if f2 in last_fight_date else None
        
        r1 = apply_multiphase_decay(r1, days_since_f1, quick_succession_days,
                                   quick_succession_bump, decay_days, decay_rate)
        r2 = apply_multiphase_decay(r2, days_since_f2, quick_succession_days,
                                   quick_succession_bump, decay_days, decay_rate)

        rating_diff = (r2 - r1) / denominator
        rating_diff = max(min(rating_diff, 100), -100)
        e1 = 1 / (1 + 10 ** rating_diff)
        e2 = 1 - e1

        if use_mov:
            mov_scale = method_of_victory_scale(row)
            k_eff = k * mov_scale
        else:
            k_eff = k

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


def evaluate_consistency(individual):
    """
    Evaluate parameters based on Sharpe ratio across rolling validation windows.
    Uses ONLY validation data for optimization.
    """
    quick_days, quick_bump, decay_days, decay_rate = individual
    
    try:
        # Load training data
        df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE').reset_index(drop=True)
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df[df['result'].notna()].copy()
        
        # Run Elo
        df_elo, ratings, _ = run_elo_with_params(
            df, k=170, base_elo=1500, denominator=400, use_mov=True,
            quick_succession_days=quick_days,
            quick_succession_bump=quick_bump,
            decay_days=decay_days,
            decay_rate=decay_rate
        )
        
        # Get validation period (last year)
        max_date = df_elo['DATE'].max()
        one_year_ago = max_date - timedelta(days=365)
        val_df = df_elo[df_elo['DATE'] > one_year_ago].copy()
        
        if len(val_df) < 100:
            return (-1000.0,)
        
        # Split into 4 quarters for rolling window analysis
        val_df['quarter'] = pd.qcut(val_df['DATE'], q=4, labels=False, duplicates='drop')
        
        window_rois = []
        
        for quarter in range(4):
            window_df = val_df[val_df['quarter'] == quarter].copy()
            
            if len(window_df) < 20:
                continue
            
            # Calculate implied ROI for this window
            window_df['prob'] = 1 / (1 + 10 ** ((window_df['opp_precomp_elo'] - window_df['precomp_elo']) / 400))
            window_df['implied_odds'] = 1 / window_df['prob']
            
            # Calculate ROI
            total_bet = len(window_df)
            winnings = (window_df['result'] * window_df['implied_odds']).sum()
            roi = ((winnings - total_bet) / total_bet) * 100
            
            window_rois.append(roi)
        
        if len(window_rois) < 2:
            return (-1000.0,)
        
        # Calculate Sharpe ratio (reward per unit of risk)
        mean_roi = np.mean(window_rois)
        std_roi = np.std(window_rois)
        
        if std_roi > 0:
            sharpe = mean_roi / std_roi
        else:
            sharpe = mean_roi
        
        # Fitness = Sharpe ratio (prioritize consistency)
        # Add small bonus for positive mean ROI
        fitness = sharpe + (0.1 * max(mean_roi, 0))
        
        return (fitness,)
        
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return (-1000.0,)


def main():
    print("=" * 80)
    print("CONSISTENCY-FOCUSED DECAY PARAMETER OPTIMIZATION")
    print("=" * 80)
    print("\nObjective: Maximize Sharpe ratio (mean ROI / std ROI) across validation windows")
    print("Method: Rolling window analysis on validation data only")
    print("OOS: Completely untouched - used only for final blind test\n")
    
    # Set up DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("quick_days", random.uniform, 20, 120)
    toolbox.register("quick_bump", random.uniform, 1.01, 1.15)
    toolbox.register("decay_days", random.uniform, 180, 720)
    toolbox.register("decay_rate", random.uniform, 0.0001, 0.01)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.quick_days, toolbox.quick_bump, 
                     toolbox.decay_days, toolbox.decay_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_consistency)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    POPULATION_SIZE = 20
    GENERATIONS = 15
    CXPB, MUTPB = 0.7, 0.3
    
    print(f"GA Configuration: Pop={POPULATION_SIZE}, Gen={GENERATIONS}, Cx={CXPB}, Mut={MUTPB}\n")
    
    pop = toolbox.population(n=POPULATION_SIZE)
    
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print(f"Initial best fitness: {max([ind.fitness.values[0] for ind in pop]):.4f}\n")
    
    best_individuals = []
    
    for gen in range(GENERATIONS):
        print(f"\n{'='*80}")
        print(f"GENERATION {gen + 1}/{GENERATIONS}")
        print(f"{'='*80}")
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                mutant[0] = max(20, min(120, mutant[0]))
                mutant[1] = max(1.01, min(1.15, mutant[1]))
                mutant[2] = max(180, min(720, mutant[2]))
                mutant[3] = max(0.0001, min(0.01, mutant[3]))
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop[:] = offspring
        
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        
        print(f"\nBest Sharpe-based fitness: {best_fitness:.4f}")
        print(f"Parameters: days={best_ind[0]:.1f}, bump={best_ind[1]:.4f}, decay_days={best_ind[2]:.1f}, rate={best_ind[3]:.6f}")
        
        best_individuals.append({
            'generation': gen + 1,
            'fitness': best_fitness,
            'quick_succession_days': best_ind[0],
            'quick_succession_bump': best_ind[1],
            'decay_days': best_ind[2],
            'decay_rate': best_ind[3]
        })
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}\n")
    
    best_ever = max(best_individuals, key=lambda x: x['fitness'])
    print(f"Best parameters (by Sharpe ratio):")
    print(f"  Quick succession days: {best_ever['quick_succession_days']:.1f}")
    print(f"  Quick succession bump: {best_ever['quick_succession_bump']:.4f}x")
    print(f"  Decay days: {best_ever['decay_days']:.1f}")
    print(f"  Decay rate: {best_ever['decay_rate']:.6f}")
    print(f"  Fitness (Sharpe): {best_ever['fitness']:.4f}\n")
    
    # Evaluate on validation and OOS for reporting
    print("Calculating final validation and OOS metrics...")
    
    df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    df_elo, ratings, _ = run_elo_with_params(
        df, k=170, base_elo=1500, denominator=400, use_mov=True,
        quick_succession_days=best_ever['quick_succession_days'],
        quick_succession_bump=best_ever['quick_succession_bump'],
        decay_days=best_ever['decay_days'],
        decay_rate=best_ever['decay_rate']
    )
    
    # Validation metrics
    max_date = df_elo['DATE'].max()
    one_year_ago = max_date - timedelta(days=365)
    val_df = df_elo[df_elo['DATE'] > one_year_ago].copy()
    val_df['prob'] = 1 / (1 + 10 ** ((val_df['opp_precomp_elo'] - val_df['precomp_elo']) / 400))
    val_df['implied_odds'] = 1 / val_df['prob']
    
    val_roi = ((val_df['result'] * val_df['implied_odds']).sum() - len(val_df)) / len(val_df) * 100
    val_acc = ((val_df['result'] == (val_df['prob'] > 0.5).astype(int)).sum() / len(val_df) * 100)
    
    print(f"\nValidation Performance:")
    print(f"  ROI: {val_roi:.2f}%")
    print(f"  Accuracy: {val_acc:.2f}%")
    print(f"  Fights: {len(val_df)}")
    
    # OOS metrics
    try:
        oos_df = pd.read_csv('data/past3_events.csv', low_memory=False)
        oos_df['DATE'] = pd.to_datetime(oos_df['DATE'])
        oos_df['result'] = pd.to_numeric(oos_df['result'], errors='coerce')
        oos_df = oos_df[oos_df['result'].notna()].copy()
        
        oos_results = []
        for _, row in oos_df.iterrows():
            f1, f2 = row['FIGHTER'], row['opp_FIGHTER']
            if f1 not in ratings or f2 not in ratings or pd.isna(row.get('avg_odds')):
                continue
            
            r1, r2 = ratings[f1], ratings[f2]
            prob = 1 / (1 + 10 ** ((r2 - r1) / 400))
            market_odds = american_odds_to_decimal(row['avg_odds'])
            
            if market_odds is None:
                continue
            
            oos_results.append({
                'prob': prob,
                'actual': row['result'],
                'market_odds': market_odds
            })
        
        if oos_results:
            oos_roi = sum((r['market_odds'] - 1) if r['actual'] == 1 else -1 for r in oos_results) / len(oos_results) * 100
            oos_acc = sum(1 for r in oos_results if (r['prob'] > 0.5) == r['actual']) / len(oos_results) * 100
            
            print(f"\nOut-of-Sample Performance (BLIND TEST):")
            print(f"  ROI: {oos_roi:.2f}%")
            print(f"  Accuracy: {oos_acc:.2f}%")
            print(f"  Fights: {len(oos_results)}")
            print(f"\n  Validation-OOS gap: {abs(val_roi - oos_roi):.2f}%")
        else:
            oos_roi, oos_acc = None, None
            print("\nOOS data not available")
    except Exception as e:
        print(f"\nCould not evaluate OOS: {e}")
        oos_roi, oos_acc = None, None
    
    # Save results
    results = {
        'optimization_type': 'consistency_focused_sharpe',
        'best_parameters': best_ever,
        'validation_performance': {'roi': float(val_roi), 'accuracy': float(val_acc), 'n_fights': len(val_df)},
        'oos_performance': {'roi': float(oos_roi) if oos_roi else None, 'accuracy': float(oos_acc) if oos_acc else None, 'n_fights': len(oos_results) if oos_results else 0},
        'generation_history': best_individuals
    }
    
    with open('decay_ga_consistency_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Results saved to: decay_ga_consistency_results.json")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
