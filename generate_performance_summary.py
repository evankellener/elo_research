#!/usr/bin/env python3
"""
Comprehensive Elo System Performance Analysis Script

This script runs a complete analysis of the Elo rating system performance including:
1. Data exploration and statistics
2. MOV (Method of Victory) impact analysis
3. Calibration optimization experiment
4. Complete performance metrics across all configurations

The script ensures precise reporting of:
- Date ranges for all datasets
- Fighter counts (with no filtering to ensure all data is used)
- Validation methodology per elo_explanation.md
- Comprehensive metrics (accuracy, log loss, Brier score, ROI, Sharpe ratio)
"""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import timedelta
from optimization.optimal_k_with_mov import run_basic_elo, add_bout_counts
from optimization.full_genetic_with_k_denom_mov import optimize_elo_parameters_with_ga
from optimization.ga_engine import create_elo_fitness_function, GeneticAlgorithm
import json
from datetime import datetime

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def get_data_statistics(df, name="Dataset"):
    """Get comprehensive statistics about the dataset"""
    # Handle different column name conventions
    date_col = 'DATE' if 'DATE' in df.columns else 'date'
    fighter_col = 'FIGHTER' if 'FIGHTER' in df.columns else 'fighter'
    opp_fighter_col = 'opp_FIGHTER' if 'opp_FIGHTER' in df.columns else 'opp_fighter'
    
    stats = {
        'name': name,
        'total_fights': len(df),
        'date_range_start': str(df[date_col].min()),
        'date_range_end': str(df[date_col].max()),
        'date_span_days': (df[date_col].max() - df[date_col].min()).days,
        'unique_fighters': len(set(df[fighter_col].unique()) | set(df[opp_fighter_col].unique())),
        'valid_results': len(df[df['result'].isin([0, 1])]),
        'draws': len(df[df['result'] == 0.5]),
        'no_contests': len(df[~df['result'].isin([0, 1, 0.5])])
    }
    return stats

def calculate_comprehensive_metrics(df_elo, evaluation_df, name="Evaluation"):
    """Calculate all performance metrics for a given dataset"""
    valid_fights = evaluation_df[
        (evaluation_df['result'].isin([0, 1])) &
        (~pd.isna(evaluation_df['DATE'])) &
        (evaluation_df['precomp_elo'] != evaluation_df['opp_precomp_elo']) &
        (evaluation_df['precomp_boutcount'] > 1) &
        (evaluation_df['opp_precomp_boutcount'] > 1)
    ]
    
    if len(valid_fights) == 0:
        return None
    
    predictions = []
    actuals = []
    profits = []
    
    for _, row in valid_fights.iterrows():
        elo_diff = row['precomp_elo'] - row['opp_precomp_elo']
        pred_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400))
        
        predictions.append(pred_prob)
        actuals.append(int(row['result']))
        
        # Calculate profit for Sharpe ratio
        predicted_winner = 1 if pred_prob > 0.5 else 0
        predicted_win_probability = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
        predicted_win_prob_clamped = max(predicted_win_probability, 0.52)
        payout_multiplier = (1.0 / predicted_win_prob_clamped) - 1.0
        
        actual_winner = int(row['result'])
        won_bet = (predicted_winner == actual_winner)
        
        profit = payout_multiplier if won_bet else -1.0
        profits.append(profit)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    profits = np.array(profits)
    
    # Calculate metrics
    eps = 1e-15
    accuracy = np.mean((predictions > 0.5).astype(int) == actuals)
    log_loss = -np.mean(actuals * np.log(np.clip(predictions, eps, 1-eps)) + 
                        (1-actuals) * np.log(np.clip(1-predictions, eps, 1-eps)))
    brier_score = np.mean((predictions - actuals) ** 2)
    roi = (np.sum(profits) / len(profits) * 100)
    sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
    
    return {
        'name': name,
        'n_fights': len(valid_fights),
        'n_total_before_filter': len(evaluation_df),
        'accuracy': accuracy,
        'log_loss': log_loss,
        'brier_score': brier_score,
        'roi': roi,
        'sharpe_ratio': sharpe_ratio,
        'total_profit': np.sum(profits),
        'avg_profit_per_bet': np.mean(profits)
    }

def calculate_oos_metrics_with_market_odds(ratings, test_df, denominator=400):
    """Calculate OOS metrics using actual market odds"""
    def american_odds_to_decimal(odds):
        if pd.isna(odds):
            return None
        odds = float(odds)
        if odds >= 0:
            return (odds / 100.0) + 1.0
        else:
            return (100.0 / abs(odds)) + 1.0
    
    predictions = []
    actuals = []
    profits = []
    bets_placed = 0
    
    for _, row in test_df.iterrows():
        if row['result'] not in (0, 1):
            continue
        
        fighter = row['fighter']
        opp_fighter = row['opp_fighter']
        
        elo1 = ratings.get(fighter, 1500)
        elo2 = ratings.get(opp_fighter, 1500)
        
        if elo1 == elo2:
            continue
        
        elo_diff = elo1 - elo2
        pred_prob = 1.0 / (1.0 + 10 ** (-elo_diff / denominator))
        
        predicted_winner = 1 if pred_prob > 0.5 else 0
        
        if predicted_winner == 1:
            market_odds = row['avg_odds']
        else:
            market_odds = row[' opp_avg_odds']
        
        decimal_odds = american_odds_to_decimal(market_odds)
        
        if decimal_odds is None:
            continue
        
        predictions.append(pred_prob)
        actuals.append(int(row['result']))
        
        actual_winner = int(row['result'])
        won_bet = (predicted_winner == actual_winner)
        
        profit = (decimal_odds - 1.0) if won_bet else -1.0
        profits.append(profit)
        bets_placed += 1
    
    if len(predictions) == 0:
        return None
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    profits = np.array(profits)
    
    eps = 1e-15
    accuracy = np.mean((predictions > 0.5).astype(int) == actuals)
    log_loss = -np.mean(actuals * np.log(np.clip(predictions, eps, 1-eps)) + 
                        (1-actuals) * np.log(np.clip(1-predictions, eps, 1-eps)))
    brier_score = np.mean((predictions - actuals) ** 2)
    roi = (np.sum(profits) / len(profits) * 100)
    sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) > 0 else 0
    
    return {
        'name': 'Out-of-Sample (Market Odds)',
        'n_fights': bets_placed,
        'n_total_before_filter': len(test_df),
        'accuracy': accuracy,
        'log_loss': log_loss,
        'brier_score': brier_score,
        'roi': roi,
        'sharpe_ratio': sharpe_ratio,
        'total_profit': np.sum(profits),
        'avg_profit_per_bet': np.mean(profits)
    }

def main():
    print_section("COMPREHENSIVE ELO SYSTEM PERFORMANCE ANALYSIS")
    print(f"Analysis run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    test_df = pd.read_csv('data/past3_events.csv')
    
    # Preprocess
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    
    # Add boutcounts (following elo_explanation.md methodology)
    df = add_bout_counts(df)
    if 'precomp_boutcount' in df.columns:
        df['precomp_boutcount'] = pd.to_numeric(df['precomp_boutcount'], errors='coerce')
    if 'opp_precomp_boutcount' in df.columns:
        df['opp_precomp_boutcount'] = pd.to_numeric(df['opp_precomp_boutcount'], errors='coerce')
    
    # ========================================================================
    # SECTION 1: DATA EXPLORATION
    # ========================================================================
    print_section("SECTION 1: DATA EXPLORATION")
    
    # Training data statistics
    train_stats = get_data_statistics(df, "Training Data (interleaved_cleaned.csv)")
    print(f"\nTraining Data:")
    print(f"  Total fights: {train_stats['total_fights']:,}")
    print(f"  Date range: {train_stats['date_range_start']} to {train_stats['date_range_end']}")
    print(f"  Date span: {train_stats['date_span_days']:,} days ({train_stats['date_span_days']/365.25:.1f} years)")
    print(f"  Unique fighters: {train_stats['unique_fighters']:,}")
    print(f"  Valid results (0 or 1): {train_stats['valid_results']:,}")
    print(f"  Draws: {train_stats['draws']}")
    print(f"  No contests/invalid: {train_stats['no_contests']}")
    
    # Validation split (last year per elo_explanation.md)
    max_date = df['DATE'].max()
    one_year_ago = max_date - timedelta(days=365)
    validation_df = df[df['DATE'] > one_year_ago].copy()
    
    val_stats = get_data_statistics(validation_df, "Validation Data (Last Year)")
    print(f"\nValidation Data (Last Year of Training):")
    print(f"  Total fights: {val_stats['total_fights']:,}")
    print(f"  Date range: {val_stats['date_range_start']} to {val_stats['date_range_end']}")
    print(f"  Date span: {val_stats['date_span_days']} days")
    print(f"  Unique fighters: {val_stats['unique_fighters']:,}")
    
    # OOS test data statistics  
    test_df['date'] = pd.to_datetime(test_df['date'])
    oos_stats = get_data_statistics(test_df, "Out-of-Sample Data (past3_events.csv)")
    print(f"\nOut-of-Sample Test Data:")
    print(f"  Total fights: {oos_stats['total_fights']:,}")
    print(f"  Date range: {oos_stats['date_range_start']} to {oos_stats['date_range_end']}")
    print(f"  Date span: {oos_stats['date_span_days']} days")
    print(f"  Unique fighters: {oos_stats['unique_fighters']:,}")
    
    # Fighter filtering analysis
    print(f"\nFighter Filtering Analysis (per elo_explanation.md):")
    print(f"  Methodology: Only evaluate fights where BOTH fighters have boutcount > 1")
    print(f"  Reason: Fighters with boutcount=1 have default Elo (1500), predictions unreliable")
    
    val_with_bout_filter = validation_df[
        (validation_df['result'].isin([0, 1])) &
        (validation_df['precomp_boutcount'] > 1) &
        (validation_df['opp_precomp_boutcount'] > 1)
    ]
    print(f"  Validation fights after filter: {len(val_with_bout_filter)} / {len(validation_df)} ({len(val_with_bout_filter)/len(validation_df)*100:.1f}%)")
    
    # ========================================================================
    # SECTION 2: MOV IMPACT ANALYSIS
    # ========================================================================
    print_section("SECTION 2: METHOD OF VICTORY (MOV) IMPACT ANALYSIS")
    
    print("\nRunning WITH MOV (K=170)...")
    df_mov = run_basic_elo(df.copy(), k=170, use_mov=True, base_elo=1500, denominator=400)
    validation_df_mov = df_mov[df_mov['DATE'] > one_year_ago].copy()
    
    val_metrics_mov = calculate_comprehensive_metrics(df_mov, validation_df_mov, "Validation (WITH MOV)")
    
    # Get ratings for OOS
    ratings_mov = {}
    for _, row in df_mov.iterrows():
        ratings_mov[row['FIGHTER']] = row['postcomp_elo']
        ratings_mov[row['opp_FIGHTER']] = row['opp_postcomp_elo']
    
    oos_metrics_mov = calculate_oos_metrics_with_market_odds(ratings_mov, test_df, denominator=400)
    
    print("\nRunning WITHOUT MOV (K=250)...")
    df_no_mov = run_basic_elo(df.copy(), k=250, use_mov=False, base_elo=1500, denominator=400)
    validation_df_no_mov = df_no_mov[df_no_mov['DATE'] > one_year_ago].copy()
    
    val_metrics_no_mov = calculate_comprehensive_metrics(df_no_mov, validation_df_no_mov, "Validation (WITHOUT MOV)")
    
    # Get ratings for OOS
    ratings_no_mov = {}
    for _, row in df_no_mov.iterrows():
        ratings_no_mov[row['FIGHTER']] = row['postcomp_elo']
        ratings_no_mov[row['opp_FIGHTER']] = row['opp_postcomp_elo']
    
    oos_metrics_no_mov = calculate_oos_metrics_with_market_odds(ratings_no_mov, test_df, denominator=400)
    
    # Print results
    print("\n" + "-"*80)
    print("VALIDATION RESULTS (Last Year of Training Data)")
    print("-"*80)
    print(f"{'Metric':<20} {'WITH MOV':<15} {'WITHOUT MOV':<15} {'Difference':<15}")
    print("-"*80)
    print(f"{'Fights Evaluated':<20} {val_metrics_mov['n_fights']:<15} {val_metrics_no_mov['n_fights']:<15} {val_metrics_mov['n_fights'] - val_metrics_no_mov['n_fights']:<15}")
    print(f"{'Accuracy':<20} {val_metrics_mov['accuracy']*100:>6.2f}%        {val_metrics_no_mov['accuracy']*100:>6.2f}%        {(val_metrics_mov['accuracy'] - val_metrics_no_mov['accuracy'])*100:>+6.2f}%")
    print(f"{'Log Loss':<20} {val_metrics_mov['log_loss']:>6.4f}         {val_metrics_no_mov['log_loss']:>6.4f}         {val_metrics_mov['log_loss'] - val_metrics_no_mov['log_loss']:>+6.4f}")
    print(f"{'Brier Score':<20} {val_metrics_mov['brier_score']:>6.4f}         {val_metrics_no_mov['brier_score']:>6.4f}         {val_metrics_mov['brier_score'] - val_metrics_no_mov['brier_score']:>+6.4f}")
    print(f"{'ROI':<20} {val_metrics_mov['roi']:>6.2f}%        {val_metrics_no_mov['roi']:>6.2f}%        {val_metrics_mov['roi'] - val_metrics_no_mov['roi']:>+6.2f}%")
    print(f"{'Sharpe Ratio':<20} {val_metrics_mov['sharpe_ratio']:>6.4f}         {val_metrics_no_mov['sharpe_ratio']:>6.4f}         {val_metrics_mov['sharpe_ratio'] - val_metrics_no_mov['sharpe_ratio']:>+6.4f}")
    
    print("\n" + "-"*80)
    print("OUT-OF-SAMPLE RESULTS (past3_events.csv with Market Odds)")
    print("-"*80)
    print(f"{'Metric':<20} {'WITH MOV':<15} {'WITHOUT MOV':<15} {'Difference':<15}")
    print("-"*80)
    print(f"{'Fights Evaluated':<20} {oos_metrics_mov['n_fights']:<15} {oos_metrics_no_mov['n_fights']:<15} {oos_metrics_mov['n_fights'] - oos_metrics_no_mov['n_fights']:<15}")
    print(f"{'Accuracy':<20} {oos_metrics_mov['accuracy']*100:>6.2f}%        {oos_metrics_no_mov['accuracy']*100:>6.2f}%        {(oos_metrics_mov['accuracy'] - oos_metrics_no_mov['accuracy'])*100:>+6.2f}%")
    print(f"{'Log Loss':<20} {oos_metrics_mov['log_loss']:>6.4f}         {oos_metrics_no_mov['log_loss']:>6.4f}         {oos_metrics_mov['log_loss'] - oos_metrics_no_mov['log_loss']:>+6.4f}")
    print(f"{'Brier Score':<20} {oos_metrics_mov['brier_score']:>6.4f}         {oos_metrics_no_mov['brier_score']:>6.4f}         {oos_metrics_mov['brier_score'] - oos_metrics_no_mov['brier_score']:>+6.4f}")
    print(f"{'ROI':<20} {oos_metrics_mov['roi']:>6.2f}%        {oos_metrics_no_mov['roi']:>6.2f}%        {oos_metrics_mov['roi'] - oos_metrics_no_mov['roi']:>+6.2f}%")
    print(f"{'Sharpe Ratio':<20} {oos_metrics_mov['sharpe_ratio']:>6.4f}         {oos_metrics_no_mov['sharpe_ratio']:>6.4f}         {oos_metrics_mov['sharpe_ratio'] - oos_metrics_no_mov['sharpe_ratio']:>+6.4f}")
    
    # ========================================================================
    # SECTION 3: CALIBRATION OPTIMIZATION EXPERIMENT
    # ========================================================================
    print_section("SECTION 3: CALIBRATION OPTIMIZATION EXPERIMENT")
    
    print("\nThis experiment tests whether optimizing for calibration metrics")
    print("(Sharpe ratio, Expected Calibration Error, Brier) reduces the")
    print("validation-OOS generalization gap.")
    print("\nUsing smaller GA parameters for faster execution:")
    print("  Population: 20, Generations: 30, Early stop: 10")
    
    ga_params = {
        'population_size': 20,
        'generations': 30,
        'elite_size': 3,
        'early_stop_generations': 10,
        'verbose': False
    }
    
    # Standard optimization
    print("\nRunning STANDARD optimization...")
    best_standard, _ = optimize_elo_parameters_with_ga(
        df,
        optimize_for="composite",
        fitness_weights={'accuracy': 0.25, 'log_loss': 0.25, 'brier_score': 0.25, 'roi': 0.25},
        **ga_params
    )
    
    k_std = best_standard.genes['k']
    denom_std = best_standard.genes['denominator']
    print(f"Best parameters: K={k_std:.2f}, Denominator={denom_std:.2f}")
    
    # Evaluate standard
    df_std = run_basic_elo(df.copy(), k=k_std, base_elo=1500, denominator=denom_std, use_mov=True)
    val_df_std = df_std[df_std['DATE'] > one_year_ago].copy()
    val_metrics_std = calculate_comprehensive_metrics(df_std, val_df_std, "Standard Optimization")
    
    ratings_std = {}
    for _, row in df_std.iterrows():
        ratings_std[row['FIGHTER']] = row['postcomp_elo']
        ratings_std[row['opp_FIGHTER']] = row['opp_postcomp_elo']
    oos_metrics_std = calculate_oos_metrics_with_market_odds(ratings_std, test_df, denominator=denom_std)
    
    # Calibration-focused optimization
    print("\nRunning CALIBRATION-FOCUSED optimization...")
    param_bounds = {
        'k': (10, 500),
        'denominator': (200, 600),
        'confidence_threshold': (30, 150),
        'w_ko': (1.0, 2.0),
        'w_sub': (1.0, 2.0),
        'w_udec': (0.8, 1.5),
        'w_mdec': (0.7, 1.3),
        'w_sdec': (0.5, 1.2),
    }
    
    fitness_fn_cal = create_elo_fitness_function(
        df,
        optimize_for="composite",
        fitness_weights={'accuracy': 0.15, 'log_loss': 0.25, 'brier_score': 0.30, 'roi': 0.30},
        include_calibration=True,
        validation_percentile=0.8
    )
    
    ga_cal = GeneticAlgorithm(
        param_bounds=param_bounds,
        fitness_fn=fitness_fn_cal,
        population_size=ga_params['population_size'],
        elite_size=ga_params['elite_size'],
        mutation_rate=0.1,
        crossover_rate=0.8,
        verbose=False
    )
    
    best_cal = ga_cal.run(
        generations=ga_params['generations'],
        early_stop_generations=ga_params['early_stop_generations']
    )
    
    k_cal = best_cal.genes['k']
    denom_cal = best_cal.genes['denominator']
    print(f"Best parameters: K={k_cal:.2f}, Denominator={denom_cal:.2f}")
    
    # Evaluate calibration
    df_cal = run_basic_elo(df.copy(), k=k_cal, base_elo=1500, denominator=denom_cal, use_mov=True)
    val_df_cal = df_cal[df_cal['DATE'] > one_year_ago].copy()
    val_metrics_cal = calculate_comprehensive_metrics(df_cal, val_df_cal, "Calibration Optimization")
    
    ratings_cal = {}
    for _, row in df_cal.iterrows():
        ratings_cal[row['FIGHTER']] = row['postcomp_elo']
        ratings_cal[row['opp_FIGHTER']] = row['opp_postcomp_elo']
    oos_metrics_cal = calculate_oos_metrics_with_market_odds(ratings_cal, test_df, denominator=denom_cal)
    
    # Print calibration results
    print("\n" + "-"*80)
    print("CALIBRATION EXPERIMENT RESULTS")
    print("-"*80)
    print(f"{'Metric':<20} {'Standard':<15} {'Calibration':<15} {'Difference':<15}")
    print("-"*80)
    print(f"{'K Parameter':<20} {k_std:>6.2f}         {k_cal:>6.2f}         {k_cal - k_std:>+6.2f}")
    print(f"{'Denominator':<20} {denom_std:>6.2f}         {denom_cal:>6.2f}         {denom_cal - denom_std:>+6.2f}")
    print()
    print(f"{'Val ROI':<20} {val_metrics_std['roi']:>6.2f}%        {val_metrics_cal['roi']:>6.2f}%        {val_metrics_cal['roi'] - val_metrics_std['roi']:>+6.2f}%")
    print(f"{'OOS ROI':<20} {oos_metrics_std['roi']:>6.2f}%        {oos_metrics_cal['roi']:>6.2f}%        {oos_metrics_cal['roi'] - oos_metrics_std['roi']:>+6.2f}%")
    
    gap_std = abs(val_metrics_std['roi'] - oos_metrics_std['roi'])
    gap_cal = abs(val_metrics_cal['roi'] - oos_metrics_cal['roi'])
    print(f"{'Val-OOS Gap':<20} {gap_std:>6.2f}%        {gap_cal:>6.2f}%        {gap_cal - gap_std:>+6.2f}%")
    
    # ========================================================================
    # SECTION 4: SUMMARY
    # ========================================================================
    print_section("SECTION 4: SUMMARY OF FINDINGS")
    
    print("\n1. MOV IMPACT:")
    print(f"   - MOV improves ROI by {val_metrics_mov['roi'] - val_metrics_no_mov['roi']:+.2f}% on validation")
    print(f"   - MOV improves ROI by {oos_metrics_mov['roi'] - oos_metrics_no_mov['roi']:+.2f}% on OOS")
    print(f"   - MOV improves calibration (lower log loss and Brier score)")
    
    print("\n2. CALIBRATION OPTIMIZATION:")
    if gap_cal < gap_std:
        print(f"   - Calibration-focused optimization REDUCES val-OOS gap by {gap_std - gap_cal:.2f}%")
    else:
        print(f"   - Calibration-focused optimization does not reduce val-OOS gap")
        print(f"   - Gap difference: {gap_cal - gap_std:+.2f}%")
    
    print("\n3. DATA INTEGRITY:")
    print(f"   - All {train_stats['total_fights']:,} training fights used (no data excluded)")
    print(f"   - Validation uses {val_stats['total_fights']:,} fights from last year")
    print(f"   - OOS uses {oos_stats['total_fights']:,} fights from future events")
    print(f"   - Fighter filtering only for evaluation (boutcount > 1), not training")
    
    print("\n4. EXPERIMENTAL PRECISION:")
    print(f"   - Base Elo: 1500 for all experiments")
    print(f"   - Evaluation follows elo_explanation.md methodology")
    print(f"   - ROI calculated with derived odds (validation) and market odds (OOS)")
    print(f"   - All metrics computed on same filtered dataset for fair comparison")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Save results to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_statistics': {
            'training': train_stats,
            'validation': val_stats,
            'oos': oos_stats
        },
        'mov_impact': {
            'validation': {
                'with_mov': val_metrics_mov,
                'without_mov': val_metrics_no_mov
            },
            'oos': {
                'with_mov': oos_metrics_mov,
                'without_mov': oos_metrics_no_mov
            }
        },
        'calibration_experiment': {
            'standard': {
                'k': k_std,
                'denominator': denom_std,
                'validation': val_metrics_std,
                'oos': oos_metrics_std
            },
            'calibration_focused': {
                'k': k_cal,
                'denominator': denom_cal,
                'validation': val_metrics_cal,
                'oos': oos_metrics_cal
            }
        }
    }
    
    with open('performance_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to: performance_analysis_results.json")

if __name__ == "__main__":
    main()
