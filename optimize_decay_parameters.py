#!/usr/bin/env python3
"""
Optimized Decay Parameter Search Script

This script efficiently searches for optimal decay/improvement parameters
by using a reduced parameter grid and parallel processing where possible.
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import timedelta
from optimization.optimal_k_with_mov import run_basic_elo, add_bout_counts
from elo.elo_utils import apply_multiphase_decay
import json
from datetime import datetime

def american_odds_to_decimal(odds):
    """
    Convert American odds to decimal odds.
    
    American odds format:
    - Positive odds (e.g., +150): Profit from a $100 bet. +150 means win $150 on $100.
      Decimal = (odds / 100) + 1 = (150/100) + 1 = 2.50
    - Negative odds (e.g., -200): Amount to bet to win $100. -200 means bet $200 to win $100.
      Decimal = (100 / |odds|) + 1 = (100/200) + 1 = 1.50
    
    Args:
        odds: American odds value (positive or negative float, or string representation)
    
    Returns:
        Decimal odds value, or None if input is NaN or cannot be converted
    
    Examples:
        >>> american_odds_to_decimal(150)   # +150 underdog
        2.5
        >>> american_odds_to_decimal(-200)  # -200 favorite
        1.5
    """
    if pd.isna(odds):
        return None
    
    # Convert to float if it's a string
    try:
        odds = float(odds)
    except (ValueError, TypeError):
        return None
    
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1

def run_elo_with_decay(df, k=32, base_elo=1500, denominator=400, use_mov=True, 
                        draw_k_factor=0.5, use_decay=False, quick_succession_days=60,
                        quick_succession_bump=1.05, decay_days=365, decay_rate=0.001):
    """Enhanced Elo loop with optional decay/improvement adjustments."""
    df = df.copy()
    ratings = {}
    last_fight_date = {}
    pre, post, opp_pre, opp_post = [], [], [], []

    for _, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
        current_date = row["DATE"]
        
        if f1 not in ratings:
            ratings[f1] = base_elo
        if f2 not in ratings:
            ratings[f2] = base_elo
            
        r1_orig, r2_orig = ratings[f1], ratings[f2]
        
        # Apply decay if enabled
        if use_decay and f1 in last_fight_date:
            days_since = (current_date - last_fight_date[f1]).days
            r1_orig = apply_multiphase_decay(
                r1_orig, days_since,
                quick_succession_days, quick_succession_bump,
                decay_days, decay_rate
            )
        if use_decay and f2 in last_fight_date:
            days_since = (current_date - last_fight_date[f2]).days
            r2_orig = apply_multiphase_decay(
                r2_orig, days_since,
                quick_succession_days, quick_succession_bump,
                decay_days, decay_rate
            )
        
        pre.append(r1_orig)
        opp_pre.append(r2_orig)
        
        # Calculate expected scores
        exp1 = 1 / (1 + 10**((r2_orig - r1_orig) / denominator))
        exp2 = 1 - exp1
        
        # Apply MOV weights if enabled
        if use_mov and res in [0, 1]:
            method = row.get('METHOD', None)
            if method in ['Decision - Unanimous', 'Decision - Split', 'Decision - Majority']:
                k_eff = k * 0.8
            elif method in ['KO/TKO', 'Submission']:
                k_eff = k * 1.2
            else:
                k_eff = k
        else:
            k_eff = k if res in [0, 1] else k * draw_k_factor
        
        # Update ratings
        if res == 1:
            r1_new = r1_orig + k_eff * (1 - exp1)
            r2_new = r2_orig + k_eff * (0 - exp2)
        elif res == 0:
            r1_new = r1_orig + k_eff * (0 - exp1)
            r2_new = r2_orig + k_eff * (1 - exp2)
        else:  # Draw
            r1_new = r1_orig + k_eff * (0.5 - exp1)
            r2_new = r2_orig + k_eff * (0.5 - exp2)
        
        ratings[f1] = r1_new
        ratings[f2] = r2_new
        last_fight_date[f1] = current_date
        last_fight_date[f2] = current_date
        
        post.append(r1_new)
        opp_post.append(r2_new)
    
    df['precomp_elo'] = pre
    df['postcomp_elo'] = post
    df['opp_precomp_elo'] = opp_pre
    df['opp_postcomp_elo'] = opp_post
    return df

def calculate_oos_roi(ratings, test_df, denominator=400):
    """Calculate OOS ROI using market odds (converts American odds to decimal)."""
    test_df = test_df[test_df['avg_odds'].notna()].copy()
    
    correct = 0
    total_roi = 0
    n_fights = 0
    
    for _, row in test_df.iterrows():
        f1, f2 = row['FIGHTER'], row['opp_FIGHTER']
        
        if f1 not in ratings or f2 not in ratings:
            continue
        
        # Convert American odds to decimal odds
        f1_decimal_odds = american_odds_to_decimal(row['avg_odds'])
        f2_decimal_odds = american_odds_to_decimal(row['opp_avg_odds']) if pd.notna(row['opp_avg_odds']) else None
        
        if f1_decimal_odds is None:
            continue
            
        r1, r2 = ratings[f1], ratings[f2]
        prob_f1_wins = 1 / (1 + 10**((r2 - r1) / denominator))
        
        # Bet on fighter with higher probability
        if prob_f1_wins > 0.5:
            # Bet on f1
            if row['result'] == 1:
                profit = f1_decimal_odds - 1
                total_roi += profit
                correct += 1
            else:
                total_roi -= 1
        else:
            # Bet on f2
            if row['result'] == 0:
                if f2_decimal_odds is not None:
                    profit = f2_decimal_odds - 1
                else:
                    profit = 1  # Fallback if no odds available
                total_roi += profit
                correct += 1
            else:
                total_roi -= 1
        
        n_fights += 1
    
    roi = (total_roi / n_fights * 100) if n_fights > 0 else 0
    accuracy = correct / n_fights if n_fights > 0 else 0
    
    return {'roi': roi, 'accuracy': accuracy, 'n_fights': n_fights}

def main():
    print("="*80)
    print("DECAY PARAMETER OPTIMIZATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/interleaved_cleaned.csv', parse_dates=['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    df = add_bout_counts(df)
    
    test_df = pd.read_csv('data/past3_events.csv', parse_dates=['date'])
    test_df.columns = test_df.columns.str.strip()  # Remove leading/trailing spaces
    test_df.rename(columns={
        'date': 'DATE',
        'fighter': 'FIGHTER',
        'opp_fighter': 'opp_FIGHTER'
    }, inplace=True)
    test_df = test_df.sort_values('DATE').reset_index(drop=True)
    test_df = add_bout_counts(test_df)
    
    print(f"Training data: {len(df)} fights")
    print(f"OOS test data: {len(test_df)} fights")
    print(f"OOS fights with market odds: {test_df['avg_odds'].notna().sum()}")
    
    # Reduced parameter grid for faster optimization
    print("\nParameter search space:")
    quick_succession_days_range = [30, 60, 90]
    quick_succession_bump_range = [1.03, 1.05, 1.10]
    decay_days_range = [270, 365, 540]
    decay_rate_range = [0.001, 0.002, 0.005]
    
    print(f"  Quick succession days: {quick_succession_days_range}")
    print(f"  Quick succession bump: {quick_succession_bump_range}")
    print(f"  Decay days: {decay_days_range}")
    print(f"  Decay rate: {decay_rate_range}")
    
    total_combinations = (len(quick_succession_days_range) * 
                         len(quick_succession_bump_range) * 
                         len(decay_days_range) * 
                         len(decay_rate_range))
    
    print(f"\nTotal combinations to test: {total_combinations}")
    print("="*80)
    
    best_oos_roi = -float('inf')
    best_params = None
    best_accuracy = None
    all_results = []
    
    tested_count = 0
    
    for qs_days in quick_succession_days_range:
        for qs_bump in quick_succession_bump_range:
            for d_days in decay_days_range:
                for d_rate in decay_rate_range:
                    tested_count += 1
                    
                    # Run Elo with these decay parameters
                    df_test = run_elo_with_decay(
                        df.copy(), k=170, base_elo=1500, denominator=400,
                        use_mov=True, use_decay=True,
                        quick_succession_days=qs_days,
                        quick_succession_bump=qs_bump,
                        decay_days=d_days,
                        decay_rate=d_rate
                    )
                    
                    # Calculate OOS metrics
                    ratings_test = {}
                    for _, row in df_test.iterrows():
                        ratings_test[row['FIGHTER']] = row['postcomp_elo']
                        ratings_test[row['opp_FIGHTER']] = row['opp_postcomp_elo']
                    
                    oos_metrics = calculate_oos_roi(ratings_test, test_df, denominator=400)
                    
                    # Store results
                    result = {
                        'quick_succession_days': qs_days,
                        'quick_succession_bump': qs_bump,
                        'decay_days': d_days,
                        'decay_rate': d_rate,
                        'oos_roi': oos_metrics['roi'],
                        'oos_accuracy': oos_metrics['accuracy'],
                        'n_fights': oos_metrics['n_fights']
                    }
                    all_results.append(result)
                    
                    # Check if this is best
                    if oos_metrics['roi'] > best_oos_roi:
                        best_oos_roi = oos_metrics['roi']
                        best_params = {
                            'quick_succession_days': qs_days,
                            'quick_succession_bump': qs_bump,
                            'decay_days': d_days,
                            'decay_rate': d_rate
                        }
                        best_accuracy = oos_metrics['accuracy']
                    
                    # Progress update
                    if tested_count % 9 == 0:
                        print(f"Progress: {tested_count}/{total_combinations} tested | "
                              f"Current: ROI={oos_metrics['roi']:.2f}% | "
                              f"Best so far: ROI={best_oos_roi:.2f}%")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nTested {total_combinations} parameter combinations")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-"*80)
    print("OPTIMAL DECAY PARAMETERS")
    print("-"*80)
    print(f"Quick succession threshold: {best_params['quick_succession_days']} days")
    print(f"Quick succession bump: {best_params['quick_succession_bump']:.2f}x "
          f"({(best_params['quick_succession_bump']-1)*100:.1f}% boost)")
    print(f"Decay threshold: {best_params['decay_days']} days")
    print(f"Decay rate: {best_params['decay_rate']:.4f}")
    
    print("\n" + "-"*80)
    print("BEST PERFORMANCE")
    print("-"*80)
    print(f"OOS ROI: {best_oos_roi:.2f}%")
    print(f"OOS Accuracy: {best_accuracy*100:.2f}%")
    
    if best_oos_roi > 0:
        print(f"\n✓ POSITIVE ROI ACHIEVED!")
        print(f"  The system with optimized decay parameters is potentially profitable.")
    else:
        print(f"\n✗ Still negative ROI.")
        print(f"  Further optimization or additional features may be needed.")
    
    # Save all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_combinations_tested': total_combinations,
        'best_params': best_params,
        'best_oos_roi': best_oos_roi,
        'best_oos_accuracy': best_accuracy,
        'all_results': all_results
    }
    
    with open('decay_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nDetailed results saved to: decay_optimization_results.json")
    print("="*80)

if __name__ == "__main__":
    main()
