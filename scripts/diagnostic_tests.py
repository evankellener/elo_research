"""
Diagnostic Test Suite for ROI Performance Analysis

This module implements 6 diagnostic tests to analyze the ROI performance degradation
between training and out-of-sample (OOS) periods.

Tests:
1. Rolling OOS Backtest - Train on past windows, test on next window
2. Parameter Robustness - Compare best params vs random params on OOS
3. Reshuffle OOS Test Set - Bootstrap ROI from random historical samples
4. Market Shift Analysis - Compare odds/rating distributions between periods
5. Bootstrap/Monte Carlo - Build ROI variance distribution
6. Accuracy vs ROI Analysis - Compare prediction accuracy and ROI

Usage:
    python scripts/diagnostic_tests.py
    python scripts/diagnostic_tests.py --test 1  # Run specific test
    python scripts/diagnostic_tests.py --all     # Run all tests
"""

import argparse
import os
import sys
import random
import math
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from full_genetic_with_k_denom_mov import (
    run_basic_elo,
    evaluate_params_roi,
    calculate_oos_roi,
    ga_search_params_roi,
    random_params,
    PARAM_BOUNDS,
    build_bidirectional_odds_lookup,
    american_odds_to_decimal,
)
from elo_utils import add_bout_counts, build_fighter_history, has_prior_history


def load_data():
    """Load and prepare all necessary data files."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load training data
    df = pd.read_csv(os.path.join(project_root, "data/interleaved_cleaned.csv"), low_memory=False)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
    df = df.sort_values("DATE").reset_index(drop=True)
    df = add_bout_counts(df)
    
    # Load OOS test data
    test_df = pd.read_csv(os.path.join(project_root, "data/past3_events.csv"), low_memory=False)
    
    # Load odds data
    odds_df = pd.read_csv(os.path.join(project_root, "after_averaging.csv"), low_memory=False)
    odds_df["DATE"] = pd.to_datetime(odds_df["DATE"]).dt.tz_localize(None)
    # Convert odds columns to numeric
    odds_df["avg_odds"] = pd.to_numeric(odds_df["avg_odds"], errors="coerce")
    
    # Prepare test odds
    test_odds_df = test_df.copy()
    test_odds_df = test_odds_df.rename(columns={
        'date': 'DATE',
        'fighter': 'FIGHTER',
        'opp_fighter': 'opp_FIGHTER'
    })
    test_odds_df["DATE"] = pd.to_datetime(test_odds_df["DATE"]).dt.tz_localize(None)
    
    return df, test_df, odds_df, test_odds_df


def get_best_params():
    """Return the best GA parameters found (from previous optimization)."""
    # These are representative best params from the GA optimization
    return {
        "k": 266.33,
        "w_ko": 1.02,
        "w_sub": 1.74,
        "w_udec": 0.98,
        "w_sdec": 0.70,
        "w_mdec": 0.81,
    }


# =============================================================================
# Test 1: Rolling OOS Backtest
# =============================================================================

def test_rolling_oos_backtest(df, odds_df, window_days=90, step_days=30, n_windows=10, verbose=True):
    """
    Test 1: Rolling OOS Backtest
    
    For each window in historical data:
    - Train GA on everything BEFORE that window
    - Test best parameters on that window
    - Compare in-sample vs OOS ROI
    
    Args:
        df: Training data
        odds_df: Odds data
        window_days: Size of each test window in days
        step_days: Step size between windows
        n_windows: Number of windows to test
        verbose: Print progress
    
    Returns:
        DataFrame with results for each window
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 1: Rolling OOS Backtest")
        print("="*60)
        print(f"Window size: {window_days} days, Step: {step_days} days, N windows: {n_windows}")
    
    max_date = df["DATE"].max()
    min_date = df["DATE"].min()
    
    results = []
    
    for i in range(n_windows):
        # Define window boundaries
        window_end = max_date - timedelta(days=i * step_days)
        window_start = window_end - timedelta(days=window_days)
        
        if window_start < min_date + timedelta(days=365):
            # Not enough training data
            break
        
        # Split data
        train_df = df[df["DATE"] < window_start].copy()
        test_window_df = df[(df["DATE"] >= window_start) & (df["DATE"] < window_end)].copy()
        
        if len(train_df) < 100 or len(test_window_df) < 10:
            continue
        
        # Get odds for the test window
        window_odds = odds_df[(odds_df["DATE"] >= window_start) & (odds_df["DATE"] < window_end)].copy()
        
        if len(window_odds) < 5:
            continue
        
        if verbose:
            print(f"\nWindow {i+1}: {window_start.date()} to {window_end.date()}")
            print(f"  Train fights: {len(train_df)}, Test fights: {len(test_window_df)}")
        
        # Quick optimization on training data (reduced for speed)
        best_params, train_roi = ga_search_params_roi(
            train_df, odds_df[odds_df["DATE"] < window_start],
            population_size=10,
            generations=5,
            lookback_days=90,
            seed=42 + i,
            verbose=False
        )
        
        # Calculate ROI on test window using trained params
        test_roi = evaluate_params_roi(
            pd.concat([train_df, test_window_df]).sort_values("DATE").reset_index(drop=True),
            window_odds,
            best_params,
            lookback_days=window_days
        )
        
        results.append({
            'window_num': i + 1,
            'window_start': window_start,
            'window_end': window_end,
            'train_fights': len(train_df),
            'test_fights': len(test_window_df),
            'train_roi': train_roi,
            'test_roi': test_roi,
            'roi_drop': train_roi - test_roi,
            'best_k': best_params['k']
        })
        
        if verbose:
            print(f"  Train ROI: {train_roi:+.2f}%, Test ROI: {test_roi:+.2f}%, Drop: {train_roi - test_roi:.2f}%")
    
    results_df = pd.DataFrame(results)
    
    if verbose and len(results_df) > 0:
        print(f"\n--- Summary ---")
        print(f"Average Train ROI: {results_df['train_roi'].mean():.2f}%")
        print(f"Average Test ROI: {results_df['test_roi'].mean():.2f}%")
        print(f"Average ROI Drop: {results_df['roi_drop'].mean():.2f}%")
        print(f"Std Dev of ROI Drop: {results_df['roi_drop'].std():.2f}%")
    
    return results_df


# =============================================================================
# Test 2: Parameter Robustness
# =============================================================================

def test_parameter_robustness(df, test_df, odds_df, test_odds_df, n_random=10, verbose=True):
    """
    Test 2: Parameter Robustness
    
    Compare best GA parameters against random parameter sets on OOS data.
    If only best params drop off → overfitting signal
    If all params drop similarly → market/data quality issue
    
    Args:
        df: Training data
        test_df: OOS test data
        odds_df: Training odds
        test_odds_df: OOS test odds
        n_random: Number of random parameter sets to test
        verbose: Print progress
    
    Returns:
        DataFrame with results for each parameter set
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 2: Parameter Robustness")
        print("="*60)
    
    best_params = get_best_params()
    
    # Get training cutoff date
    test_start_date = pd.to_datetime(test_df["date"]).min()
    train_df = df[df["DATE"] < test_start_date].copy()
    train_odds = odds_df[odds_df["DATE"] < test_start_date].copy()
    
    results = []
    
    # Test best params
    train_roi = evaluate_params_roi(train_df, train_odds, best_params, lookback_days=90)
    
    # Train Elo and calculate OOS ROI
    mov_params = {k: v for k, v in best_params.items() if k.startswith('w_')}
    df_trained = run_basic_elo(train_df.copy(), k=best_params['k'], mov_params=mov_params)
    oos_result = calculate_oos_roi(df_trained, test_df, test_odds_df, verbose=False)
    
    results.append({
        'param_type': 'best',
        'k': best_params['k'],
        'train_roi': train_roi,
        'oos_roi': oos_result['roi_percent'],
        'roi_drop': train_roi - oos_result['roi_percent'],
        'oos_bets': oos_result['total_bets']
    })
    
    if verbose:
        print(f"Best params: Train ROI={train_roi:+.2f}%, OOS ROI={oos_result['roi_percent']:+.2f}%")
    
    # Test random params
    random.seed(42)
    for i in range(n_random):
        rand_params = random_params()
        
        train_roi = evaluate_params_roi(train_df, train_odds, rand_params, lookback_days=90)
        
        mov_params = {k: v for k, v in rand_params.items() if k.startswith('w_')}
        df_trained = run_basic_elo(train_df.copy(), k=rand_params['k'], mov_params=mov_params)
        oos_result = calculate_oos_roi(df_trained, test_df, test_odds_df, verbose=False)
        
        results.append({
            'param_type': f'random_{i+1}',
            'k': rand_params['k'],
            'train_roi': train_roi,
            'oos_roi': oos_result['roi_percent'],
            'roi_drop': train_roi - oos_result['roi_percent'],
            'oos_bets': oos_result['total_bets']
        })
        
        if verbose:
            print(f"Random {i+1}: Train ROI={train_roi:+.2f}%, OOS ROI={oos_result['roi_percent']:+.2f}%")
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n--- Summary ---")
        best_drop = results_df[results_df['param_type'] == 'best']['roi_drop'].iloc[0]
        random_drops = results_df[results_df['param_type'] != 'best']['roi_drop']
        print(f"Best params ROI drop: {best_drop:.2f}%")
        print(f"Random params avg drop: {random_drops.mean():.2f}% (std: {random_drops.std():.2f}%)")
        
        if best_drop > random_drops.mean() + random_drops.std():
            print("⚠️ OVERFITTING SIGNAL: Best params drop more than random params")
        else:
            print("✓ No clear overfitting signal - drops are similar across param sets")
    
    return results_df


# =============================================================================
# Test 3: Reshuffle OOS Test Set
# =============================================================================

def test_reshuffle_oos(df, odds_df, oos_size=36, n_samples=100, verbose=True):
    """
    Test 3: Reshuffle OOS Test Set
    
    Compare current OOS ROI to ROI from random historical fight samples.
    See if current OOS fights are abnormally unprofitable.
    
    Args:
        df: Training data
        odds_df: Odds data
        oos_size: Number of fights per sample (matching OOS size)
        n_samples: Number of bootstrap samples
        verbose: Print progress
    
    Returns:
        dict with bootstrap statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 3: Reshuffle OOS Test Set (Bootstrap)")
        print("="*60)
        print(f"Sample size: {oos_size} fights, N samples: {n_samples}")
    
    best_params = get_best_params()
    
    # Get unique events/dates with odds
    odds_dates = odds_df["DATE"].unique()
    
    # Calculate ROI for each random sample
    bootstrap_rois = []
    random.seed(42)
    
    for i in range(n_samples):
        # Sample random dates
        if len(odds_dates) < oos_size // 3:
            continue
        
        sample_dates = np.random.choice(odds_dates, size=min(oos_size // 3, len(odds_dates)), replace=False)
        sample_odds = odds_df[odds_df["DATE"].isin(sample_dates)].copy()
        
        if len(sample_odds) < 10:
            continue
        
        # Calculate ROI on this sample
        roi = evaluate_params_roi(df, sample_odds, best_params, lookback_days=0)
        bootstrap_rois.append(roi)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_samples} samples...")
    
    bootstrap_rois = np.array(bootstrap_rois)
    
    # Calculate statistics
    mean_roi = np.mean(bootstrap_rois)
    std_roi = np.std(bootstrap_rois)
    percentile_5 = np.percentile(bootstrap_rois, 5)
    percentile_95 = np.percentile(bootstrap_rois, 95)
    
    # Current OOS ROI for comparison (approximate)
    current_oos_roi = -11.11  # From problem statement
    
    # Calculate z-score
    z_score = (current_oos_roi - mean_roi) / std_roi if std_roi > 0 else 0
    
    results = {
        'mean_roi': mean_roi,
        'std_roi': std_roi,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'current_oos_roi': current_oos_roi,
        'z_score': z_score,
        'bootstrap_rois': bootstrap_rois
    }
    
    if verbose:
        print(f"\n--- Bootstrap Results ---")
        print(f"Mean ROI: {mean_roi:.2f}%")
        print(f"Std Dev: {std_roi:.2f}%")
        print(f"90% CI: [{percentile_5:.2f}%, {percentile_95:.2f}%]")
        print(f"Current OOS ROI: {current_oos_roi:.2f}%")
        print(f"Z-score: {z_score:.2f}")
        
        if z_score < -2:
            print("⚠️ Current OOS is significantly below bootstrap distribution (p < 0.05)")
        elif z_score < -1:
            print("⚠️ Current OOS is below average but within normal range")
        else:
            print("✓ Current OOS is within normal variance range")
    
    return results


# =============================================================================
# Test 4: Market Shift Analysis
# =============================================================================

def test_market_shift(df, test_df, odds_df, test_odds_df, verbose=True):
    """
    Test 4: Market Shift Analysis
    
    Compare odds distributions, fighter rating distributions between 
    training and OOS periods. Check for systematic differences.
    
    Args:
        df: Training data
        test_df: OOS test data
        odds_df: Training odds
        test_odds_df: OOS test odds
        verbose: Print progress
    
    Returns:
        dict with distribution statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 4: Market Shift Analysis")
        print("="*60)
    
    # Get training cutoff
    test_start_date = pd.to_datetime(test_df["date"]).min()
    
    # Split odds data
    train_odds = odds_df[odds_df["DATE"] < test_start_date].copy()
    
    # Analyze odds distributions - convert to numeric first
    train_odds_values = pd.to_numeric(train_odds["avg_odds"], errors="coerce").dropna()
    if "avg_odds" in test_odds_df.columns:
        test_odds_values = pd.to_numeric(test_odds_df["avg_odds"], errors="coerce").dropna()
    else:
        test_odds_values = pd.Series([], dtype=float)
    
    results = {
        'train_period': {
            'start': df["DATE"].min(),
            'end': test_start_date,
            'n_fights': len(df[df["DATE"] < test_start_date])
        },
        'oos_period': {
            'start': test_start_date,
            'end': pd.to_datetime(test_df["date"]).max(),
            'n_fights': len(test_df)
        }
    }
    
    if len(train_odds_values) > 0:
        results['train_odds'] = {
            'mean': train_odds_values.mean(),
            'std': train_odds_values.std(),
            'median': train_odds_values.median(),
            'pct_favorites': (train_odds_values < 0).mean() * 100
        }
    
    if len(test_odds_values) > 0:
        results['oos_odds'] = {
            'mean': test_odds_values.mean(),
            'std': test_odds_values.std(),
            'median': test_odds_values.median(),
            'pct_favorites': (test_odds_values < 0).mean() * 100
        }
    
    # Analyze Elo distributions (last 90 days of training)
    # Need to run Elo first to get precomp_elo
    best_params = get_best_params()
    mov_params = {k: v for k, v in best_params.items() if k.startswith('w_')}
    df_with_elo = run_basic_elo(df.copy(), k=best_params['k'], mov_params=mov_params)
    
    recent_train = df_with_elo[df_with_elo["DATE"] >= (test_start_date - timedelta(days=90))].copy()
    if len(recent_train) > 0 and "precomp_elo" in recent_train.columns:
        elo_diffs = abs(recent_train["precomp_elo"] - recent_train["opp_precomp_elo"]).dropna()
        if len(elo_diffs) > 0:
            results['train_elo_diff'] = {
                'mean': elo_diffs.mean(),
                'std': elo_diffs.std(),
                'median': elo_diffs.median()
            }
    
    if verbose:
        print(f"\nTraining Period: {results['train_period']['start'].date()} to {results['train_period']['end'].date()}")
        print(f"  Fights: {results['train_period']['n_fights']}")
        
        print(f"\nOOS Period: {results['oos_period']['start'].date()} to {results['oos_period']['end'].date()}")
        print(f"  Fights: {results['oos_period']['n_fights']}")
        
        if 'train_odds' in results:
            print(f"\nTraining Odds Distribution:")
            print(f"  Mean: {results['train_odds']['mean']:.1f}")
            print(f"  Std: {results['train_odds']['std']:.1f}")
            print(f"  % Favorites: {results['train_odds']['pct_favorites']:.1f}%")
        
        if 'oos_odds' in results:
            print(f"\nOOS Odds Distribution:")
            print(f"  Mean: {results['oos_odds']['mean']:.1f}")
            print(f"  Std: {results['oos_odds']['std']:.1f}")
            print(f"  % Favorites: {results['oos_odds']['pct_favorites']:.1f}%")
        
        if 'train_elo_diff' in results:
            print(f"\nRecent Training Elo Differences:")
            print(f"  Mean: {results['train_elo_diff']['mean']:.1f}")
            print(f"  Median: {results['train_elo_diff']['median']:.1f}")
    
    return results


# =============================================================================
# Test 5: Bootstrap/Monte Carlo
# =============================================================================

def test_bootstrap_monte_carlo(df, odds_df, window_size=36, n_iterations=1000, verbose=True):
    """
    Test 5: Bootstrap/Monte Carlo
    
    Resample random fight windows from historical data.
    Build distribution of expected ROI variance.
    
    Args:
        df: Training data
        odds_df: Odds data
        window_size: Number of fights per window
        n_iterations: Number of bootstrap iterations
        verbose: Print progress
    
    Returns:
        dict with Monte Carlo statistics
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 5: Bootstrap/Monte Carlo Simulation")
        print("="*60)
        print(f"Window size: {window_size} fights, Iterations: {n_iterations}")
    
    best_params = get_best_params()
    
    # Build odds lookup
    odds_lookup = build_bidirectional_odds_lookup(odds_df)
    
    # Build fighter history for filtering
    mov_params = {k: v for k, v in best_params.items() if k.startswith('w_')}
    df_with_elo = run_basic_elo(df.copy(), k=best_params['k'], mov_params=mov_params)
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    # Get fights with odds available
    eligible_fights = []
    processed = set()
    
    for _, row in df_with_elo.iterrows():
        if row["result"] not in (0, 1):
            continue
        if pd.isna(row["DATE"]):
            continue
        
        fighter = row["FIGHTER"]
        opponent = row["opp_FIGHTER"]
        date_str = str(row["DATE"].date())
        
        fight_key = tuple(sorted([fighter, opponent])) + (date_str,)
        if fight_key in processed:
            continue
        processed.add(fight_key)
        
        # Check prior history
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        # Check odds availability
        if row["precomp_elo"] > row["opp_precomp_elo"]:
            odds_key = (fighter, opponent, date_str)
        else:
            odds_key = (opponent, fighter, date_str)
        
        if odds_key in odds_lookup:
            eligible_fights.append({
                'fighter': fighter,
                'opponent': opponent,
                'date': row["DATE"],
                'date_str': date_str,
                'result': row["result"],
                'fighter_elo': row["precomp_elo"],
                'opp_elo': row["opp_precomp_elo"],
                'odds_key': odds_key
            })
    
    if verbose:
        print(f"Eligible fights with odds: {len(eligible_fights)}")
    
    if len(eligible_fights) < window_size:
        if verbose:
            print("Not enough eligible fights for Monte Carlo")
        return {'error': 'insufficient_data'}
    
    # Run Monte Carlo
    random.seed(42)
    sample_rois = []
    
    for i in range(n_iterations):
        # Sample random fights
        sample = random.sample(eligible_fights, window_size)
        
        # Calculate ROI for this sample
        total_wagered = 0.0
        total_profit = 0.0
        
        for fight in sample:
            # Determine bet
            if fight['fighter_elo'] > fight['opp_elo']:
                bet_won = (fight['result'] == 1)
            else:
                bet_won = (fight['result'] == 0)
            
            odds = odds_lookup[fight['odds_key']]
            decimal_odds = american_odds_to_decimal(odds)
            
            if decimal_odds is None:
                continue
            
            bet_amount = 1.0
            total_wagered += bet_amount
            
            if bet_won:
                total_profit += (bet_amount * decimal_odds) - bet_amount
            else:
                total_profit -= bet_amount
        
        if total_wagered > 0:
            roi = (total_profit / total_wagered) * 100
            sample_rois.append(roi)
        
        if verbose and (i + 1) % 200 == 0:
            print(f"  Completed {i+1}/{n_iterations} iterations...")
    
    sample_rois = np.array(sample_rois)
    
    # Calculate statistics
    mean_roi = np.mean(sample_rois)
    std_roi = np.std(sample_rois)
    percentile_5 = np.percentile(sample_rois, 5)
    percentile_95 = np.percentile(sample_rois, 95)
    min_roi = np.min(sample_rois)
    max_roi = np.max(sample_rois)
    
    # The observed swing is from +12.29% to -11.11%
    observed_swing = 12.29 - (-11.11)  # = 23.4%
    
    # Calculate how often we see swings this large
    swing_threshold = observed_swing
    swing_count = 0
    for i in range(len(sample_rois)):
        for j in range(i + 1, len(sample_rois)):
            if abs(sample_rois[i] - sample_rois[j]) >= swing_threshold:
                swing_count += 1
    
    total_pairs = len(sample_rois) * (len(sample_rois) - 1) // 2
    swing_pct = (swing_count / total_pairs * 100) if total_pairs > 0 else 0
    
    results = {
        'mean_roi': mean_roi,
        'std_roi': std_roi,
        'percentile_5': percentile_5,
        'percentile_95': percentile_95,
        'min_roi': min_roi,
        'max_roi': max_roi,
        'observed_swing': observed_swing,
        'swing_probability': swing_pct,
        'sample_rois': sample_rois
    }
    
    if verbose:
        print(f"\n--- Monte Carlo Results ---")
        print(f"Mean ROI: {mean_roi:.2f}%")
        print(f"Std Dev: {std_roi:.2f}%")
        print(f"Range: [{min_roi:.2f}%, {max_roi:.2f}%]")
        print(f"90% CI: [{percentile_5:.2f}%, {percentile_95:.2f}%]")
        print(f"\nObserved swing (+12% → -11%): {observed_swing:.2f}%")
        print(f"Expected range (2 std): {2 * std_roi:.2f}%")
        
        if observed_swing <= 2 * std_roi:
            print("✓ Observed swing is within expected variance (2 std)")
        else:
            print("⚠️ Observed swing exceeds expected variance")
    
    return results


# =============================================================================
# Test 6: Accuracy vs ROI Analysis
# =============================================================================

def test_accuracy_vs_roi(df, test_df, odds_df, test_odds_df, verbose=True):
    """
    Test 6: Accuracy vs ROI Analysis
    
    Compare prediction accuracy between training and OOS.
    Check if predictions are correct but market odds don't allow profit.
    
    Args:
        df: Training data
        test_df: OOS test data
        odds_df: Training odds
        test_odds_df: OOS test odds
        verbose: Print progress
    
    Returns:
        dict with accuracy and ROI metrics
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST 6: Accuracy vs ROI Analysis")
        print("="*60)
    
    best_params = get_best_params()
    
    # Get training cutoff
    test_start_date = pd.to_datetime(test_df["date"]).min()
    train_df = df[df["DATE"] < test_start_date].copy()
    train_odds = odds_df[odds_df["DATE"] < test_start_date].copy()
    
    # Train Elo
    mov_params = {k: v for k, v in best_params.items() if k.startswith('w_')}
    df_trained = run_basic_elo(train_df.copy(), k=best_params['k'], mov_params=mov_params)
    
    # Get extended metrics for training period
    train_extended = evaluate_params_roi(
        train_df, train_odds, best_params, 
        lookback_days=90, return_extended=True
    )
    
    # Get OOS metrics
    oos_result = calculate_oos_roi(df_trained, test_df, test_odds_df, verbose=False)
    
    # Calculate implied accuracy from ROI
    # If accuracy = 60% and avg odds = -150 (1.67 decimal), expected ROI = 0.6 * 1.67 - 1 = 0.002 = 0.2%
    # Break-even accuracy for -150 odds = 1 / 1.67 = 59.9%
    
    results = {
        'training': {
            'accuracy': train_extended.get('accuracy'),
            'roi': train_extended.get('roi_percent'),
            'num_bets': train_extended.get('num_bets'),
            'win_rate': train_extended.get('win_rate'),
            'log_loss': train_extended.get('log_loss'),
            'brier_score': train_extended.get('brier_score')
        },
        'oos': {
            'accuracy': oos_result.get('accuracy'),
            'roi': oos_result.get('roi_percent'),
            'num_bets': oos_result.get('total_bets'),
            'win_rate': oos_result.get('accuracy')  # Same as accuracy in this context
        }
    }
    
    if verbose:
        print("\n--- Training Period (Last 90 Days) ---")
        if results['training']['accuracy']:
            print(f"Accuracy: {results['training']['accuracy']*100:.2f}%")
        if results['training']['roi']:
            print(f"ROI: {results['training']['roi']:.2f}%")
        if results['training']['num_bets']:
            print(f"Number of Bets: {results['training']['num_bets']}")
        if results['training']['win_rate']:
            print(f"Win Rate: {results['training']['win_rate']*100:.2f}%")
        
        print("\n--- OOS Period ---")
        if results['oos']['accuracy']:
            print(f"Accuracy: {results['oos']['accuracy']*100:.2f}%")
        if results['oos']['roi'] is not None:
            print(f"ROI: {results['oos']['roi']:.2f}%")
        if results['oos']['num_bets']:
            print(f"Number of Bets: {results['oos']['num_bets']}")
        
        # Analysis
        print("\n--- Analysis ---")
        train_acc = results['training']['accuracy']
        oos_acc = results['oos']['accuracy']
        
        if train_acc and oos_acc:
            acc_drop = (train_acc - oos_acc) * 100
            print(f"Accuracy Drop: {acc_drop:.2f}%")
            
            train_roi = results['training']['roi'] or 0
            oos_roi = results['oos']['roi'] or 0
            roi_drop = train_roi - oos_roi
            print(f"ROI Drop: {roi_drop:.2f}%")
            
            # Check if accuracy drop explains ROI drop
            # Rough estimate: 1% accuracy = ~2% ROI (depends on odds)
            expected_roi_drop = acc_drop * 2
            print(f"Expected ROI Drop from Accuracy: ~{expected_roi_drop:.2f}%")
            
            if roi_drop > expected_roi_drop + 5:
                print("⚠️ ROI drop is larger than accuracy drop suggests")
                print("   This may indicate worse odds on OOS fights")
            else:
                print("✓ ROI drop is consistent with accuracy drop")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def create_diagnostic_plots(results, output_dir=None):
    """Create visualization plots for diagnostic results."""
    if output_dir is None:
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(output_dir, "images")
        os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Diagnostic Test Results: ROI Performance Analysis", fontsize=14)
    
    # Plot 1: Rolling OOS results (if available)
    ax1 = axes[0, 0]
    if 'rolling_oos' in results and len(results['rolling_oos']) > 0:
        rolling_df = results['rolling_oos']
        x = range(len(rolling_df))
        ax1.bar([i - 0.2 for i in x], rolling_df['train_roi'], width=0.4, label='Train ROI', alpha=0.7)
        ax1.bar([i + 0.2 for i in x], rolling_df['test_roi'], width=0.4, label='Test ROI', alpha=0.7)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Window')
        ax1.set_ylabel('ROI (%)')
        ax1.set_title('Test 1: Rolling OOS Backtest')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No rolling OOS data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Test 1: Rolling OOS Backtest')
    
    # Plot 2: Parameter robustness
    ax2 = axes[0, 1]
    if 'param_robustness' in results and len(results['param_robustness']) > 0:
        rob_df = results['param_robustness']
        colors = ['red' if t == 'best' else 'blue' for t in rob_df['param_type']]
        ax2.scatter(rob_df['train_roi'], rob_df['oos_roi'], c=colors, s=100, alpha=0.7)
        ax2.plot([-50, 50], [-50, 50], 'k--', alpha=0.3, label='No Drop Line')
        ax2.set_xlabel('Training ROI (%)')
        ax2.set_ylabel('OOS ROI (%)')
        ax2.set_title('Test 2: Parameter Robustness')
        ax2.legend(['No Drop Line', 'Best Params', 'Random Params'])
    else:
        ax2.text(0.5, 0.5, 'No parameter data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Test 2: Parameter Robustness')
    
    # Plot 3: Bootstrap distribution
    ax3 = axes[1, 0]
    if 'bootstrap' in results and 'bootstrap_rois' in results['bootstrap']:
        rois = results['bootstrap']['bootstrap_rois']
        ax3.hist(rois, bins=30, alpha=0.7, edgecolor='black')
        current_oos = results['bootstrap'].get('current_oos_roi', -11.11)
        ax3.axvline(x=current_oos, color='red', linestyle='--', linewidth=2, label=f'Current OOS ({current_oos:.1f}%)')
        ax3.axvline(x=np.mean(rois), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(rois):.1f}%)')
        ax3.set_xlabel('ROI (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Test 3: Bootstrap ROI Distribution')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No bootstrap data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Test 3: Bootstrap ROI Distribution')
    
    # Plot 4: Monte Carlo distribution
    ax4 = axes[1, 1]
    if 'monte_carlo' in results and 'sample_rois' in results['monte_carlo']:
        rois = results['monte_carlo']['sample_rois']
        ax4.hist(rois, bins=30, alpha=0.7, edgecolor='black')
        mean_roi = np.mean(rois)
        std_roi = np.std(rois)
        ax4.axvline(x=mean_roi, color='green', linestyle='-', linewidth=2, label=f'Mean ({mean_roi:.1f}%)')
        ax4.axvline(x=mean_roi - 2*std_roi, color='orange', linestyle='--', linewidth=1, label=f'±2 Std')
        ax4.axvline(x=mean_roi + 2*std_roi, color='orange', linestyle='--', linewidth=1)
        ax4.set_xlabel('ROI (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Test 5: Monte Carlo ROI Distribution')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'No Monte Carlo data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Test 5: Monte Carlo ROI Distribution')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "diagnostic_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to: {output_path}")
    return output_path


# =============================================================================
# Main
# =============================================================================

def run_all_tests(verbose=True):
    """Run all diagnostic tests and return results."""
    print("="*60)
    print("DIAGNOSTIC TEST SUITE")
    print("ROI Performance Degradation Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df, test_df, odds_df, test_odds_df = load_data()
    print(f"Training data: {len(df)} fights")
    print(f"OOS test data: {len(test_df)} fights")
    print(f"Odds data: {len(odds_df)} records")
    
    results = {}
    
    # Test 1: Rolling OOS Backtest
    try:
        results['rolling_oos'] = test_rolling_oos_backtest(
            df, odds_df, 
            window_days=90, step_days=45, n_windows=5,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 1 error: {e}")
        results['rolling_oos'] = pd.DataFrame()
    
    # Test 2: Parameter Robustness
    try:
        results['param_robustness'] = test_parameter_robustness(
            df, test_df, odds_df, test_odds_df,
            n_random=5,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 2 error: {e}")
        results['param_robustness'] = pd.DataFrame()
    
    # Test 3: Reshuffle OOS
    try:
        results['bootstrap'] = test_reshuffle_oos(
            df, odds_df,
            oos_size=36, n_samples=50,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 3 error: {e}")
        results['bootstrap'] = {}
    
    # Test 4: Market Shift
    try:
        results['market_shift'] = test_market_shift(
            df, test_df, odds_df, test_odds_df,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 4 error: {e}")
        results['market_shift'] = {}
    
    # Test 5: Monte Carlo
    try:
        results['monte_carlo'] = test_bootstrap_monte_carlo(
            df, odds_df,
            window_size=36, n_iterations=500,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 5 error: {e}")
        results['monte_carlo'] = {}
    
    # Test 6: Accuracy vs ROI
    try:
        results['accuracy_roi'] = test_accuracy_vs_roi(
            df, test_df, odds_df, test_odds_df,
            verbose=verbose
        )
    except Exception as e:
        print(f"Test 6 error: {e}")
        results['accuracy_roi'] = {}
    
    # Generate plots
    try:
        create_diagnostic_plots(results)
    except Exception as e:
        print(f"Plot generation error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("\nKey Findings:")
    
    if len(results.get('rolling_oos', pd.DataFrame())) > 0:
        avg_drop = results['rolling_oos']['roi_drop'].mean()
        print(f"1. Average ROI drop in rolling backtest: {avg_drop:.2f}%")
    
    if len(results.get('param_robustness', pd.DataFrame())) > 0:
        best_drop = results['param_robustness'][results['param_robustness']['param_type'] == 'best']['roi_drop'].iloc[0]
        print(f"2. Best params ROI drop: {best_drop:.2f}%")
    
    if 'monte_carlo' in results and 'std_roi' in results['monte_carlo']:
        std_roi = results['monte_carlo']['std_roi']
        print(f"5. Expected ROI variance (1 std): ±{std_roi:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic Test Suite for ROI Performance Analysis"
    )
    parser.add_argument(
        "--test", type=int, choices=[1, 2, 3, 4, 5, 6],
        help="Run specific test (1-6)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--decay-mode", choices=["linear", "exponential", "none"], default="none",
        dest="decay_mode",
        help="Decay mode for Elo ratings: 'linear', 'exponential', or 'none' (default: none)"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Load data
    df, test_df, odds_df, test_odds_df = load_data()
    
    if args.all or args.test is None:
        run_all_tests(verbose=verbose)
    elif args.test == 1:
        test_rolling_oos_backtest(df, odds_df, verbose=verbose)
    elif args.test == 2:
        test_parameter_robustness(df, test_df, odds_df, test_odds_df, verbose=verbose)
    elif args.test == 3:
        test_reshuffle_oos(df, odds_df, verbose=verbose)
    elif args.test == 4:
        test_market_shift(df, test_df, odds_df, test_odds_df, verbose=verbose)
    elif args.test == 5:
        test_bootstrap_monte_carlo(df, odds_df, verbose=verbose)
    elif args.test == 6:
        test_accuracy_vs_roi(df, test_df, odds_df, test_odds_df, verbose=verbose)


if __name__ == "__main__":
    main()
