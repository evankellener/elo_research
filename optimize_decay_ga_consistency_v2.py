#!/usr/bin/env python3
"""
Genetic Algorithm-based decay parameter optimization.
Treats decay parameters as continuous values within ranges.
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
        # Clip to avoid overflow in exponential
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


def evaluate_decay_parameters(individual):
    """
    Evaluate a set of decay parameters using GA.
    
    Parameters (individual):
    - quick_succession_days: 20-120 days
    - quick_succession_bump: 1.01-1.15
    - decay_days: 180-720 days  
    - decay_rate: 0.0001-0.01
    
    Returns tuple: (oos_roi,) for optimization
    """
    quick_days, quick_bump, decay_days, decay_rate = individual
    
    # Load and prepare data
    df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE')
    # Ensure result is numeric
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    
    oos_df = pd.read_csv('data/past3_events.csv')
    oos_df['DATE'] = pd.to_datetime(oos_df['date'])
    oos_df = oos_df.rename(columns={
        'fighter': 'FIGHTER',
        'opp_fighter': 'opp_FIGHTER'
    })
    # result column is already correct (1 for win, 0 for loss)
    oos_df['result'] = pd.to_numeric(oos_df['result'], errors='coerce')
    
    # Split into training and validation
    last_date = df['DATE'].max()
    one_year_ago = last_date - timedelta(days=365)
    
    train_df = df[df['DATE'] < one_year_ago].copy()
    val_df = df[df['DATE'] >= one_year_ago].copy()
    
    # Run Elo on training data
    train_result, ratings, last_fight_dates = run_elo_with_params(
        train_df,
        k=170,
        base_elo=1500,
        denominator=400,
        use_mov=True,
        quick_succession_days=int(quick_days),
        quick_succession_bump=quick_bump,
        decay_days=int(decay_days),
        decay_rate=decay_rate
    )
    
    # Evaluate on validation
    val_results = []
    for _, row in val_df.iterrows():
        f1, f2 = row['FIGHTER'], row['opp_FIGHTER']
        
        if f1 not in ratings or f2 not in ratings:
            continue
            
        r1 = ratings[f1]
        r2 = ratings[f2]
        
        prob = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual = row['result']
        
        val_results.append({
            'prob': prob,
            'actual': actual
        })
    
    # Calculate validation ROI (using derived odds)
    val_roi = 0
    val_correct = 0
    if len(val_results) > 0:
        for result in val_results:
            pred = 1 if result['prob'] > 0.5 else 0
            if pred == result['actual']:
                val_correct += 1
                if result['prob'] > 0.5:
                    val_roi += (1 / result['prob'] - 1)
                else:
                    val_roi += (1 / (1 - result['prob']) - 1)
            else:
                val_roi -= 1
        val_roi = (val_roi / len(val_results)) * 100
        val_accuracy = val_correct / len(val_results)
    else:
        val_roi = -100
        val_accuracy = 0
    
    # Evaluate on OOS with market odds
    oos_results = []
    for _, row in oos_df.iterrows():
        f1, f2 = row['FIGHTER'], row['opp_FIGHTER']
        
        if f1 not in ratings or f2 not in ratings:
            continue
        if pd.isna(row.get('avg_odds')):
            continue
            
        r1 = ratings[f1]
        r2 = ratings[f2]
        
        prob = 1 / (1 + 10 ** ((r2 - r1) / 400))
        actual = row['result']
        
        # Convert American odds to decimal
        market_odds = american_odds_to_decimal(row['avg_odds'])
        if market_odds is None:
            continue
        
        oos_results.append({
            'prob': prob,
            'actual': actual,
            'market_odds': market_odds
        })
    
    # Calculate OOS ROI with market odds
    oos_roi = 0
    oos_correct = 0
    if len(oos_results) > 0:
        for result in oos_results:
            pred = 1 if result['prob'] > 0.5 else 0
            if pred == result['actual']:
                oos_correct += 1
                oos_roi += (result['market_odds'] - 1)
            else:
                oos_roi -= 1
        oos_roi = (oos_roi / len(oos_results)) * 100
        oos_accuracy = oos_correct / len(oos_results)
    else:
        oos_roi = -100
        oos_accuracy = 0
    
    # Store results for printing
    evaluate_decay_parameters.last_result = {
        'params': {
            'quick_succession_days': quick_days,
            'quick_succession_bump': quick_bump,
            'decay_days': decay_days,
            'decay_rate': decay_rate
        },
        'validation': {
            'roi': val_roi,
            'accuracy': val_accuracy,
            'n_fights': len(val_results)
        },
        'oos': {
            'roi': oos_roi,
            'accuracy': oos_accuracy,
            'n_fights': len(oos_results)
        }
    }
    
    # Return tuple for DEAP: (oos_roi,)
    return (oos_roi,)

# Set up DEAP GA
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Define parameter ranges
toolbox.register("quick_days", random.uniform, 20, 120)
toolbox.register("quick_bump", random.uniform, 1.01, 1.15)
toolbox.register("decay_days", random.uniform, 180, 720)
toolbox.register("decay_rate", random.uniform, 0.0001, 0.01)

# Individual and population
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.quick_days, toolbox.quick_bump, 
                  toolbox.decay_days, toolbox.decay_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic operators
toolbox.register("evaluate", evaluate_decay_parameters)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)
    np.random.seed(42)
    
    # GA parameters
    POPULATION_SIZE = 20
    GENERATIONS = 15
    CXPB = 0.7
    MUTPB = 0.3
    
    print("Starting GA-based decay parameter optimization")
    print(f"Population: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print("Parameter Ranges:")
    print("  quick_succession_days: 20-120")
    print("  quick_succession_bump: 1.01-1.15")
    print("  decay_days: 180-720")
    print("  decay_rate: 0.0001-0.01\n")
    
    # Create initial population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Track best individual
    best_results = []
    all_results = []
    
    # Evolution
    for gen in range(GENERATIONS):
        print(f"\n=== Generation {gen + 1}/{GENERATIONS} ===")
        
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation
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
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                mutant[0] = max(20, min(120, mutant[0]))
                mutant[1] = max(1.01, min(1.15, mutant[1]))
                mutant[2] = max(180, min(720, mutant[2]))
                mutant[3] = max(0.0001, min(0.01, mutant[3]))
                del mutant.fitness.values
        
        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            all_results.append(evaluate_decay_parameters.last_result)
        
        # Replace population
        pop[:] = offspring
        
        # Get best individual of this generation
        best_ind = tools.selBest(pop, 1)[0]
        best_result = evaluate_decay_parameters.last_result
        best_results.append(best_result)
        
        print(f"Best OOS ROI: {best_result['oos']['roi']:.2f}%")
        print(f"Parameters: days={best_result['params']['quick_succession_days']:.1f}, "
              f"bump={best_result['params']['quick_succession_bump']:.3f}, "
              f"decay_days={best_result['params']['decay_days']:.1f}, "
              f"rate={best_result['params']['decay_rate']:.5f}")
        print(f"Validation: ROI={best_result['validation']['roi']:.2f}%, "
              f"Acc={best_result['validation']['accuracy']:.2%}, "
              f"N={best_result['validation']['n_fights']}")
        print(f"OOS: ROI={best_result['oos']['roi']:.2f}%, "
              f"Acc={best_result['oos']['accuracy']:.2%}, "
              f"N={best_result['oos']['n_fights']}")
    
    # Final best individual
    final_best = tools.selBest(pop, 1)[0]
    toolbox.evaluate(final_best)
    final_result = evaluate_decay_parameters.last_result
    
    print("\n\n=== OPTIMIZATION COMPLETE ===")
    print("\nBest Parameters:")
    print(f"  quick_succession_days: {final_result['params']['quick_succession_days']:.1f}")
    print(f"  quick_succession_bump: {final_result['params']['quick_succession_bump']:.4f}")
    print(f"  decay_days: {final_result['params']['decay_days']:.1f}")
    print(f"  decay_rate: {final_result['params']['decay_rate']:.6f}")
    print("\nValidation Performance:")
    print(f"  ROI: {final_result['validation']['roi']:.2f}%")
    print(f"  Accuracy: {final_result['validation']['accuracy']:.2%}")
    print(f"  Fights: {final_result['validation']['n_fights']}")
    print("\nOut-of-Sample Performance:")
    print(f"  ROI: {final_result['oos']['roi']:.2f}%")
    print(f"  Accuracy: {final_result['oos']['accuracy']:.2%}")
    print(f"  Fights: {final_result['oos']['n_fights']}")
    
    # Save results
    output = {
        'optimization_method': 'Genetic Algorithm',
        'population_size': POPULATION_SIZE,
        'generations': GENERATIONS,
        'parameter_ranges': {
            'quick_succession_days': [20, 120],
            'quick_succession_bump': [1.01, 1.15],
            'decay_days': [180, 720],
            'decay_rate': [0.0001, 0.01]
        },
        'best_parameters': final_result['params'],
        'best_performance': {
            'validation': final_result['validation'],
            'oos': final_result['oos']
        },
        'generation_history': best_results,
        'all_evaluations': all_results
    }
    
    with open('decay_ga_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to decay_ga_optimization_results.json")

if __name__ == '__main__':
    main()
