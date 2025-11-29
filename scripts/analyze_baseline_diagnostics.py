"""
Diagnostic Analysis Script for Baseline Elo Predictions

This script analyzes baseline OOS predictions to identify which subgroups have 
poor accuracy/ROI. This tells us exactly which parameters to optimize next.

Analysis Dimensions:
1. By Fighter Experience Level (Unknown, Novice, Experienced, Veteran)
2. By Layoff Duration (Recent, Medium, Long, Very Long)
3. By Fight Method (KO/TKO, Submission, Unanimous/Majority/Split Decision)
4. By Opponent Quality (Elite, Mid-tier, Lower-tier Elo)
5. By Elo Margin (Huge favorite, Large favorite, Small favorite, Close, Underdog)
6. Over Time (Monthly trends)
7. Confidence Calibration (ECE by decile)

Usage:
    python scripts/analyze_baseline_diagnostics.py
    python scripts/analyze_baseline_diagnostics.py --output-json results/diagnostics.json
    python scripts/analyze_baseline_diagnostics.py --no-visualizations
"""

import argparse
import json
import os
import sys
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from full_genetic_with_k_denom_mov import (
    run_basic_elo,
    build_bidirectional_odds_lookup,
    american_odds_to_decimal,
)
from elo_utils import add_bout_counts, build_fighter_history, has_prior_history, is_true


# =============================================================================
# Constants
# =============================================================================

# Experience Level Thresholds (prior bout counts)
EXPERIENCE_LEVELS = {
    'Unknown': (0, 5),       # 0-5 prior bouts
    'Novice': (6, 15),       # 6-15 prior bouts
    'Experienced': (16, 50), # 16-50 prior bouts
    'Veteran': (51, float('inf'))  # 51+ prior bouts
}

# Layoff Duration Thresholds (days)
LAYOFF_THRESHOLDS = {
    'Recent': (0, 90),       # 0-90 days
    'Medium': (91, 180),     # 91-180 days
    'Long': (181, 274),      # 181-274 days
    'Very Long': (275, float('inf'))  # 275+ days
}

# Opponent Quality Thresholds (pre-fight Elo)
OPPONENT_QUALITY = {
    'Lower-tier': (0, 1500),
    'Mid-tier': (1500, 1700),
    'Elite': (1700, float('inf'))
}

# Elo Margin Thresholds
ELO_MARGINS = {
    'Huge favorite (>300)': (300, float('inf')),
    'Large favorite (200-300)': (200, 300),
    'Small favorite (100-200)': (100, 200),
    'Close (0-100)': (0, 100),
    'Close underdog (0-100)': (-100, 0),
    'Large underdog (100-200)': (-200, -100),
    'Very large underdog (200+)': (float('-inf'), -200)
}

# Fight Methods
FIGHT_METHODS = ['KO/TKO', 'Submission', 'Unanimous Decision', 
                 'Majority Decision', 'Split Decision', 'Other']

# Calibration Deciles
CALIBRATION_BINS = [(i/10, (i+1)/10) for i in range(10)]


# =============================================================================
# Data Loading and Preparation
# =============================================================================

def load_data(project_root: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare all necessary data files."""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load training data
    df = pd.read_csv(os.path.join(project_root, "data/interleaved_cleaned.csv"), low_memory=False)
    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
    df = df.sort_values("DATE").reset_index(drop=True)
    df = add_bout_counts(df)
    
    # Ensure boutcount columns are numeric
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")
    
    # Load OOS test data
    test_df = pd.read_csv(os.path.join(project_root, "data/past3_events.csv"), low_memory=False)
    test_df["result"] = pd.to_numeric(test_df["result"], errors="coerce")
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.tz_localize(None)
    
    # Load odds data
    odds_df = pd.read_csv(os.path.join(project_root, "after_averaging.csv"), low_memory=False)
    odds_df["DATE"] = pd.to_datetime(odds_df["DATE"]).dt.tz_localize(None)
    odds_df["avg_odds"] = pd.to_numeric(odds_df["avg_odds"], errors="coerce")
    
    return df, test_df, odds_df


def get_baseline_params() -> Dict:
    """Return baseline Elo parameters."""
    return {
        "k": 167.19618191211478,  # Default from main.py
        "w_ko": 1.4,
        "w_sub": 1.3,
        "w_udec": 1.0,
        "w_sdec": 0.7,
        "w_mdec": 0.9,
    }


def get_method_of_victory(row: pd.Series) -> str:
    """Determine the method of victory from fight data."""
    if is_true(row.get('ko')) or is_true(row.get('kod')):
        return 'KO/TKO'
    if is_true(row.get('subw')) or is_true(row.get('subwd')):
        return 'Submission'
    if is_true(row.get('udec')) or is_true(row.get('udecd')):
        return 'Unanimous Decision'
    if is_true(row.get('sdec')) or is_true(row.get('sdecd')):
        return 'Split Decision'
    if is_true(row.get('mdec')) or is_true(row.get('mdecd')):
        return 'Majority Decision'
    return 'Other'


def get_experience_level(bout_count: int) -> str:
    """Categorize fighter by experience level based on bout count."""
    if bout_count is None or pd.isna(bout_count):
        return 'Unknown'
    bout_count = int(bout_count)
    for level, (low, high) in EXPERIENCE_LEVELS.items():
        if low <= bout_count <= high:
            return level
    # Fallback for very high bout counts (shouldn't happen with infinity bound)
    return 'Veteran'


def get_layoff_category(days: Optional[float]) -> str:
    """Categorize layoff duration."""
    if days is None or pd.isna(days):
        return 'Unknown'
    days = float(days)
    # Use exclusive upper bounds to avoid overlap
    if days <= 90:
        return 'Recent'
    elif days <= 180:
        return 'Medium'
    elif days <= 274:
        return 'Long'
    else:
        return 'Very Long'


def get_opponent_quality(elo: float) -> str:
    """Categorize opponent quality by Elo rating."""
    for quality, (low, high) in OPPONENT_QUALITY.items():
        if low <= elo < high:
            return quality
    return 'Mid-tier'


def get_elo_margin_category(margin: float) -> str:
    """Categorize Elo margin.
    
    Categories:
    - margin > 300: Huge favorite
    - 200 < margin <= 300: Large favorite
    - 100 < margin <= 200: Small favorite
    - 0 < margin <= 100: Close
    - -100 < margin <= 0: Close underdog
    - -200 < margin <= -100: Large underdog
    - margin <= -200: Very large underdog
    """
    if margin > 300:
        return 'Huge favorite (>300)'
    elif margin > 200:
        return 'Large favorite (200-300)'
    elif margin > 100:
        return 'Small favorite (100-200)'
    elif margin > 0:
        return 'Close (0-100)'
    elif margin > -100:
        return 'Close underdog (0-100)'
    elif margin > -200:
        return 'Large underdog (100-200)'
    else:
        return 'Very large underdog (200+)'


# =============================================================================
# Core Analysis Functions
# =============================================================================

def build_prediction_records(
    df: pd.DataFrame,
    odds_df: pd.DataFrame,
    params: Dict,
    lookback_days: int = 0
) -> pd.DataFrame:
    """
    Build comprehensive prediction records with all features needed for analysis.
    
    Returns a DataFrame with one row per unique fight, including:
    - Fighter info (names, bout counts, layoff days)
    - Elo info (precomp_elo, opp_precomp_elo, margin)
    - Prediction info (predicted winner, actual result, correct)
    - Betting info (odds, profit, ROI)
    - Fight info (method, date)
    """
    # Train Elo model
    mov_params = {k: v for k, v in params.items() if k.startswith('w_')}
    df_with_elo = run_basic_elo(df.copy(), k=params['k'], mov_params=mov_params)
    
    # Filter to lookback period if specified
    if lookback_days and lookback_days > 0:
        max_date = df_with_elo["DATE"].max()
        cutoff_date = max_date - pd.Timedelta(days=lookback_days)
        df_with_elo = df_with_elo[df_with_elo["DATE"] > cutoff_date].copy()
    
    # Build fighter history and odds lookup
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    odds_lookup = build_bidirectional_odds_lookup(odds_df)
    
    # Pre-compute fight dates for each fighter (sorted by date)
    # This is O(n) and avoids the O(n²) nested loop later
    fighter_fight_dates = {}
    for _, row in df.sort_values("DATE").iterrows():
        date = row["DATE"]
        fighter = row["FIGHTER"]
        opponent = row["opp_FIGHTER"]
        
        if fighter not in fighter_fight_dates:
            fighter_fight_dates[fighter] = []
        fighter_fight_dates[fighter].append(date)
        
        if opponent not in fighter_fight_dates:
            fighter_fight_dates[opponent] = []
        fighter_fight_dates[opponent].append(date)
    
    records = []
    processed_fights = set()
    
    def get_layoff_days(fighter_name: str, fight_date: pd.Timestamp) -> Optional[float]:
        """Get days since fighter's last fight before the given date."""
        if fighter_name not in fighter_fight_dates:
            return None
        dates = fighter_fight_dates[fighter_name]
        # Find the last fight date before the current fight
        last_fight = None
        for d in dates:
            if d < fight_date:
                last_fight = d
            else:
                break
        if last_fight is None:
            return None
        return (fight_date - last_fight).days
    
    for _, row in df_with_elo.iterrows():
        result = row["result"]
        if result not in (0, 1):
            continue
        if pd.isna(row["DATE"]):
            continue
        
        fighter = row["FIGHTER"]
        opponent = row["opp_FIGHTER"]
        date_str = str(row["DATE"].date())
        
        # Skip duplicates
        fight_key = tuple(sorted([fighter, opponent])) + (date_str,)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)
        
        # Skip if equal Elo or no prior history
        if row["precomp_elo"] == row["opp_precomp_elo"]:
            continue
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        # Determine prediction (higher Elo predicted to win)
        fighter_elo = row["precomp_elo"]
        opp_elo = row["opp_precomp_elo"]
        elo_diff = fighter_elo - opp_elo
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        if fighter_elo > opp_elo:
            bet_on = fighter
            bet_against = opponent
            bet_on_elo = fighter_elo
            bet_against_elo = opp_elo
            bet_on_boutcount = row.get("precomp_boutcount", 0) or 0
            bet_against_boutcount = row.get("opp_precomp_boutcount", 0) or 0
            bet_won = (result == 1)
            odds_key = (fighter, opponent, date_str)
            pred_confidence = pred_prob
        else:
            bet_on = opponent
            bet_against = fighter
            bet_on_elo = opp_elo
            bet_against_elo = fighter_elo
            bet_on_boutcount = row.get("opp_precomp_boutcount", 0) or 0
            bet_against_boutcount = row.get("precomp_boutcount", 0) or 0
            bet_won = (result == 0)
            odds_key = (opponent, fighter, date_str)
            pred_confidence = 1 - pred_prob
        
        # Get odds
        if odds_key not in odds_lookup:
            # Try reverse key
            odds_key = (odds_key[1], odds_key[0], odds_key[2])
            if odds_key not in odds_lookup:
                continue
        
        odds = odds_lookup[odds_key]
        decimal_odds = american_odds_to_decimal(odds)
        if decimal_odds is None:
            continue
        
        # Calculate profit
        bet_amount = 1.0
        if bet_won:
            profit = (bet_amount * decimal_odds) - bet_amount
        else:
            profit = -bet_amount
        
        # Calculate layoff days using pre-computed fight dates (O(n) per fighter)
        fighter_layoff = get_layoff_days(fighter, row["DATE"])
        opp_layoff = get_layoff_days(opponent, row["DATE"])
        
        # Determine which layoff to use (fighter we bet on)
        if bet_on == fighter:
            bet_on_layoff = fighter_layoff
            bet_against_layoff = opp_layoff
        else:
            bet_on_layoff = opp_layoff
            bet_against_layoff = fighter_layoff
        
        # Get fight method
        method = get_method_of_victory(row)
        
        records.append({
            'date': row["DATE"],
            'month': row["DATE"].strftime('%Y-%m'),
            'fighter': fighter,
            'opponent': opponent,
            'bet_on': bet_on,
            'bet_against': bet_against,
            'bet_on_elo': bet_on_elo,
            'bet_against_elo': bet_against_elo,
            'elo_margin': bet_on_elo - bet_against_elo,
            'bet_on_boutcount': bet_on_boutcount,
            'bet_against_boutcount': bet_against_boutcount,
            'bet_on_layoff': bet_on_layoff,
            'bet_against_layoff': bet_against_layoff,
            'pred_confidence': pred_confidence,
            'bet_won': int(bet_won),
            'american_odds': odds,
            'decimal_odds': decimal_odds,
            'bet_amount': bet_amount,
            'profit': profit,
            'method': method,
            'result': result,
            'fighter_won': int(result == 1),
            # Categorizations
            'bet_on_experience': get_experience_level(int(bet_on_boutcount)),
            'bet_against_experience': get_experience_level(int(bet_against_boutcount)),
            'bet_on_layoff_category': get_layoff_category(bet_on_layoff),
            'bet_against_layoff_category': get_layoff_category(bet_against_layoff),
            'opponent_quality': get_opponent_quality(bet_against_elo),
            'elo_margin_category': get_elo_margin_category(bet_on_elo - bet_against_elo),
        })
    
    return pd.DataFrame(records)


def calculate_metrics(records: pd.DataFrame) -> Dict:
    """Calculate accuracy, ROI, and count from prediction records."""
    if len(records) == 0:
        return {'accuracy': None, 'roi': None, 'count': 0, 'win_rate': None}
    
    accuracy = records['bet_won'].mean()
    total_wagered = records['bet_amount'].sum()
    total_profit = records['profit'].sum()
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    win_rate = records['bet_won'].sum() / len(records)
    
    return {
        'accuracy': accuracy,
        'roi': roi,
        'count': len(records),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_wagered': total_wagered
    }


# =============================================================================
# Analysis 1: By Fighter Experience Level
# =============================================================================

def analyze_by_experience(records: pd.DataFrame) -> Dict:
    """Analyze predictions by fighter experience level."""
    results = {}
    
    for level in EXPERIENCE_LEVELS.keys():
        level_records = records[records['bet_on_experience'] == level]
        metrics = calculate_metrics(level_records)
        results[level] = metrics
        
        # Flag weak spots
        if metrics['accuracy'] is not None and metrics['accuracy'] < 0.50:
            metrics['flag'] = 'WEAK: Accuracy < 50%'
        elif metrics['roi'] is not None and metrics['roi'] < 0:
            metrics['flag'] = 'WEAK: Negative ROI'
        else:
            metrics['flag'] = None
    
    # Also analyze by opponent experience
    opp_results = {}
    for level in EXPERIENCE_LEVELS.keys():
        level_records = records[records['bet_against_experience'] == level]
        opp_results[level] = calculate_metrics(level_records)
    
    # Calculate insights
    insights = []
    accuracies = {k: v['accuracy'] for k, v in results.items() if v['accuracy'] is not None}
    if accuracies:
        best_level = max(accuracies, key=accuracies.get)
        worst_level = min(accuracies, key=accuracies.get)
        if accuracies[best_level] - accuracies[worst_level] > 0.05:
            insights.append(f"Experience level matters. Best: {best_level} ({accuracies[best_level]*100:.1f}%), "
                          f"Worst: {worst_level} ({accuracies[worst_level]*100:.1f}%)")
    
    return {
        'by_bet_on_experience': results,
        'by_opponent_experience': opp_results,
        'insights': insights
    }


# =============================================================================
# Analysis 2: By Layoff Duration
# =============================================================================

def analyze_by_layoff(records: pd.DataFrame) -> Dict:
    """Analyze predictions by layoff duration."""
    results = {}
    
    for category in LAYOFF_THRESHOLDS.keys():
        cat_records = records[records['bet_on_layoff_category'] == category]
        metrics = calculate_metrics(cat_records)
        results[category] = metrics
        
        # Flag weak spots
        if metrics['accuracy'] is not None and metrics['accuracy'] < 0.50:
            metrics['flag'] = 'WEAK: Accuracy < 50%'
        elif metrics['roi'] is not None and metrics['roi'] < 0:
            metrics['flag'] = 'WEAK: Negative ROI'
        else:
            metrics['flag'] = None
    
    # Calculate insights
    insights = []
    if 'Very Long' in results and results['Very Long']['accuracy'] is not None:
        vl_acc = results['Very Long']['accuracy']
        other_accs = [v['accuracy'] for k, v in results.items() 
                      if k != 'Very Long' and v['accuracy'] is not None]
        if other_accs and vl_acc < np.mean(other_accs) - 0.05:
            insights.append(f"Predictions collapse for 275+ day layoffs! "
                          f"Very Long: {vl_acc*100:.1f}%, Others avg: {np.mean(other_accs)*100:.1f}%")
            insights.append("Recommendation: More aggressive decay OR special handling for 1+ year layoffs")
    
    return {
        'by_layoff': results,
        'insights': insights
    }


# =============================================================================
# Analysis 3: By Fight Method
# =============================================================================

def analyze_by_method(records: pd.DataFrame) -> Dict:
    """Analyze predictions by fight method."""
    results = {}
    
    for method in FIGHT_METHODS:
        method_records = records[records['method'] == method]
        metrics = calculate_metrics(method_records)
        results[method] = metrics
        
        # Flag weak/strong spots
        if metrics['accuracy'] is not None:
            if metrics['accuracy'] >= 0.65:
                metrics['flag'] = 'STRONG'
            elif metrics['accuracy'] < 0.50:
                metrics['flag'] = 'WEAK: Accuracy < 50%'
            elif metrics['roi'] is not None and metrics['roi'] < 0:
                metrics['flag'] = 'WEAK: Negative ROI'
            else:
                metrics['flag'] = None
    
    # Calculate insights
    insights = []
    valid_methods = {k: v for k, v in results.items() if v['accuracy'] is not None}
    if valid_methods:
        best_method = max(valid_methods, key=lambda k: valid_methods[k]['accuracy'])
        worst_method = min(valid_methods, key=lambda k: valid_methods[k]['accuracy'])
        
        if valid_methods[best_method]['accuracy'] - valid_methods[worst_method]['accuracy'] > 0.10:
            insights.append(f"Method weighting is working. Best: {best_method} "
                          f"({valid_methods[best_method]['accuracy']*100:.1f}%), "
                          f"Worst: {worst_method} ({valid_methods[worst_method]['accuracy']*100:.1f}%)")
        
        if 'Split Decision' in valid_methods and valid_methods['Split Decision']['accuracy'] < 0.50:
            insights.append("Split decisions are hard to predict. Consider down-weighting confidence.")
    
    return {
        'by_method': results,
        'insights': insights
    }


# =============================================================================
# Analysis 4: By Opponent Quality
# =============================================================================

def analyze_by_opponent_quality(records: pd.DataFrame) -> Dict:
    """Analyze predictions by opponent quality (Elo tier)."""
    results = {}
    
    for quality in OPPONENT_QUALITY.keys():
        quality_records = records[records['opponent_quality'] == quality]
        metrics = calculate_metrics(quality_records)
        results[quality] = metrics
        
        # Flag weak/strong spots
        if metrics['accuracy'] is not None:
            if metrics['accuracy'] >= 0.65:
                metrics['flag'] = 'STRONG'
            elif metrics['accuracy'] < 0.50:
                metrics['flag'] = 'WEAK: Accuracy < 50%'
            elif metrics['roi'] is not None and metrics['roi'] < 0:
                metrics['flag'] = 'WEAK: Negative ROI'
            else:
                metrics['flag'] = None
    
    # Calculate insights
    insights = []
    valid_qualities = {k: v for k, v in results.items() if v['accuracy'] is not None}
    if 'Elite' in valid_qualities and 'Lower-tier' in valid_qualities:
        elite_acc = valid_qualities['Elite']['accuracy']
        lower_acc = valid_qualities['Lower-tier']['accuracy']
        if elite_acc > lower_acc + 0.05:
            insights.append(f"Model is better at predicting elite matchups. "
                          f"Elite: {elite_acc*100:.1f}%, Lower-tier: {lower_acc*100:.1f}%")
            insights.append("Recommendation: Apply confidence modifiers based on opponent tier.")
    
    return {
        'by_opponent_quality': results,
        'insights': insights
    }


# =============================================================================
# Analysis 5: By Elo Margin
# =============================================================================

def analyze_by_elo_margin(records: pd.DataFrame) -> Dict:
    """Analyze predictions by Elo margin (favorite vs underdog)."""
    results = {}
    
    for category in ELO_MARGINS.keys():
        cat_records = records[records['elo_margin_category'] == category]
        metrics = calculate_metrics(cat_records)
        results[category] = metrics
        
        # Flag weak spots
        if metrics['accuracy'] is not None:
            if 'underdog' in category.lower() and metrics['roi'] is not None and metrics['roi'] < -10:
                metrics['flag'] = 'WEAK: Large negative ROI on underdogs'
            elif metrics['accuracy'] < 0.50:
                metrics['flag'] = 'WEAK: Accuracy < 50%'
            elif metrics['roi'] is not None and metrics['roi'] < 0:
                metrics['flag'] = 'WEAK: Negative ROI'
            elif metrics['accuracy'] >= 0.70:
                metrics['flag'] = 'STRONG'
            else:
                metrics['flag'] = None
    
    # Calculate insights
    insights = []
    underdog_cats = [k for k in results.keys() if 'underdog' in k.lower()]
    for cat in underdog_cats:
        if results[cat]['roi'] is not None and results[cat]['roi'] < -10:
            insights.append(f"Model underestimates large underdogs. {cat}: ROI={results[cat]['roi']:.1f}%")
            insights.append("Recommendation: Adjust expected_score_denom to widen curve OR add underdog boost.")
            break
    
    return {
        'by_elo_margin': results,
        'insights': insights
    }


# =============================================================================
# Analysis 6: Over Time (Monthly)
# =============================================================================

def analyze_over_time(records: pd.DataFrame) -> Dict:
    """Analyze predictions over time by month."""
    results = {}
    
    for month in sorted(records['month'].unique()):
        month_records = records[records['month'] == month]
        metrics = calculate_metrics(month_records)
        results[month] = metrics
    
    # Detect drift
    months = sorted(results.keys())
    if len(months) >= 3:
        recent_months = months[-3:]
        early_months = months[:3] if len(months) >= 6 else months[:len(months)//2]
        
        recent_accs = [results[m]['accuracy'] for m in recent_months if results[m]['accuracy'] is not None]
        early_accs = [results[m]['accuracy'] for m in early_months if results[m]['accuracy'] is not None]
        
        if recent_accs and early_accs:
            recent_avg = np.mean(recent_accs)
            early_avg = np.mean(early_accs)
            
            if early_avg - recent_avg > 0.05:
                for m in recent_months:
                    if results[m]['accuracy'] is not None:
                        results[m]['flag'] = 'DRIFT: Trending down'
    
    # Calculate trend
    insights = []
    accuracies = [(m, results[m]['accuracy']) for m in months if results[m]['accuracy'] is not None]
    if len(accuracies) >= 4:
        x = np.arange(len(accuracies))
        y = np.array([acc for _, acc in accuracies])
        
        # Simple linear regression
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        slope = numerator / denominator if denominator > 0 else 0
        
        if slope < -0.005:
            insights.append(f"Model accuracy/ROI drifting downward over time (slope: {slope*100:.2f}%/month).")
            insights.append("Recommendation: May need recency weighting or periodic retraining.")
        elif slope > 0.005:
            insights.append(f"Model accuracy improving over time (slope: {slope*100:.2f}%/month).")
    
    return {
        'by_month': results,
        'trend_slope': slope if len(accuracies) >= 4 else None,
        'insights': insights
    }


# =============================================================================
# Analysis 7: Confidence Calibration
# =============================================================================

def analyze_calibration(records: pd.DataFrame) -> Dict:
    """Analyze prediction calibration by confidence decile."""
    results = {}
    weighted_error_sum = 0.0
    total_samples = len(records)
    
    for i, (low, high) in enumerate(CALIBRATION_BINS):
        bin_name = f'{int(low*100)}-{int(high*100)}%'
        
        if i == len(CALIBRATION_BINS) - 1:
            # Include upper bound for last bin
            bin_records = records[(records['pred_confidence'] >= low) & (records['pred_confidence'] <= high)]
        else:
            bin_records = records[(records['pred_confidence'] >= low) & (records['pred_confidence'] < high)]
        
        n_in_bin = len(bin_records)
        if n_in_bin > 0:
            avg_pred = bin_records['pred_confidence'].mean()
            actual_rate = bin_records['bet_won'].mean()
            bin_error = abs(avg_pred - actual_rate)
            weighted_error_sum += bin_error * n_in_bin
            
            # Determine calibration status
            if abs(avg_pred - actual_rate) <= 0.05:
                status = 'CALIBRATED'
            elif actual_rate > avg_pred:
                status = 'UNDERCONFIDENT'
            else:
                status = 'OVERCONFIDENT'
        else:
            avg_pred = None
            actual_rate = None
            bin_error = None
            status = None
        
        results[bin_name] = {
            'count': n_in_bin,
            'avg_predicted': avg_pred,
            'actual_rate': actual_rate,
            'bin_error': bin_error,
            'status': status
        }
    
    # Calculate ECE
    ece = weighted_error_sum / total_samples if total_samples > 0 else None
    
    # Calculate insights
    insights = []
    overconfident_bins = [k for k, v in results.items() 
                         if v['status'] == 'OVERCONFIDENT' and v['count'] >= 5]
    underconfident_bins = [k for k, v in results.items() 
                          if v['status'] == 'UNDERCONFIDENT' and v['count'] >= 5]
    
    if overconfident_bins:
        insights.append(f"Model is overconfident in bins: {', '.join(overconfident_bins)}")
        insights.append("Recommendation: Add calibration adjustment for these confidence ranges.")
    
    if underconfident_bins:
        insights.append(f"Model is underconfident in bins: {', '.join(underconfident_bins)}")
    
    if ece is not None:
        if ece < 0.03:
            insights.append(f"ECE: {ece:.3f} (excellent calibration)")
        elif ece < 0.06:
            insights.append(f"ECE: {ece:.3f} (good calibration)")
        elif ece < 0.10:
            insights.append(f"ECE: {ece:.3f} (moderate calibration error)")
        else:
            insights.append(f"ECE: {ece:.3f} (poor calibration - needs adjustment)")
    
    return {
        'by_decile': results,
        'ece': ece,
        'insights': insights
    }


# =============================================================================
# Summary and Optimization Targets
# =============================================================================

def calculate_summary(records: pd.DataFrame, all_analyses: Dict) -> Dict:
    """Calculate overall summary metrics."""
    metrics = calculate_metrics(records)
    
    # Calculate Sharpe ratio
    if len(records) > 1:
        daily_profits = records.groupby('date')['profit'].sum()
        if len(daily_profits) > 1 and daily_profits.std() > 0:
            sharpe = (daily_profits.mean() / daily_profits.std()) * np.sqrt(52)  # Annualized
        else:
            sharpe = None
    else:
        sharpe = None
    
    return {
        'total_fights': metrics['count'],
        'accuracy': metrics['accuracy'],
        'roi': metrics['roi'],
        'sharpe': sharpe,
        'win_rate': metrics['win_rate']
    }


def identify_weak_spots(all_analyses: Dict) -> List[Dict]:
    """Identify all weak spots from analyses."""
    weak_spots = []
    
    # Check experience analysis
    if 'experience' in all_analyses:
        for level, metrics in all_analyses['experience']['by_bet_on_experience'].items():
            if metrics.get('flag') and 'WEAK' in metrics['flag']:
                weak_spots.append({
                    'category': 'Experience',
                    'subgroup': level,
                    'accuracy': metrics['accuracy'],
                    'roi': metrics['roi'],
                    'count': metrics['count'],
                    'issue': metrics['flag']
                })
    
    # Check layoff analysis
    if 'layoff' in all_analyses:
        for category, metrics in all_analyses['layoff']['by_layoff'].items():
            if metrics.get('flag') and 'WEAK' in metrics['flag']:
                weak_spots.append({
                    'category': 'Layoff',
                    'subgroup': category,
                    'accuracy': metrics['accuracy'],
                    'roi': metrics['roi'],
                    'count': metrics['count'],
                    'issue': metrics['flag']
                })
    
    # Check method analysis
    if 'method' in all_analyses:
        for method, metrics in all_analyses['method']['by_method'].items():
            if metrics.get('flag') and 'WEAK' in metrics['flag']:
                weak_spots.append({
                    'category': 'Method',
                    'subgroup': method,
                    'accuracy': metrics['accuracy'],
                    'roi': metrics['roi'],
                    'count': metrics['count'],
                    'issue': metrics['flag']
                })
    
    # Check opponent quality analysis
    if 'opponent_quality' in all_analyses:
        for quality, metrics in all_analyses['opponent_quality']['by_opponent_quality'].items():
            if metrics.get('flag') and 'WEAK' in metrics['flag']:
                weak_spots.append({
                    'category': 'Opponent Quality',
                    'subgroup': quality,
                    'accuracy': metrics['accuracy'],
                    'roi': metrics['roi'],
                    'count': metrics['count'],
                    'issue': metrics['flag']
                })
    
    # Check Elo margin analysis
    if 'elo_margin' in all_analyses:
        for category, metrics in all_analyses['elo_margin']['by_elo_margin'].items():
            if metrics.get('flag') and 'WEAK' in metrics['flag']:
                weak_spots.append({
                    'category': 'Elo Margin',
                    'subgroup': category,
                    'accuracy': metrics['accuracy'],
                    'roi': metrics['roi'],
                    'count': metrics['count'],
                    'issue': metrics['flag']
                })
    
    return weak_spots


def identify_strong_spots(all_analyses: Dict) -> List[Dict]:
    """Identify all strong spots from analyses."""
    strong_spots = []
    
    for analysis_name, analysis_data in all_analyses.items():
        for key in analysis_data.keys():
            if key.startswith('by_'):
                for subgroup, metrics in analysis_data[key].items():
                    if isinstance(metrics, dict) and metrics.get('flag') == 'STRONG':
                        strong_spots.append({
                            'category': analysis_name.replace('_', ' ').title(),
                            'subgroup': subgroup,
                            'accuracy': metrics['accuracy'],
                            'roi': metrics['roi'],
                            'count': metrics['count']
                        })
    
    return strong_spots


def calculate_optimization_targets(weak_spots: List[Dict], summary: Dict) -> List[Dict]:
    """
    Calculate potential ROI impact from fixing weak spots.
    
    Estimates how much ROI improvement we'd get from fixing each weak spot.
    """
    targets = []
    baseline_roi = summary.get('roi', 0) or 0
    
    for spot in weak_spots:
        if spot['accuracy'] is None or spot['count'] == 0:
            continue
        
        current_acc = spot['accuracy']
        current_roi = spot['roi'] if spot['roi'] is not None else 0
        count = spot['count']
        
        # Estimate target accuracy (at least 55% or match overall)
        target_acc = max(0.55, summary.get('accuracy', 0.55) or 0.55)
        
        # Estimate ROI impact
        # Rough estimate: 1% accuracy improvement = ~2% ROI improvement
        acc_improvement = target_acc - current_acc
        roi_improvement = acc_improvement * 2 * 100  # Convert to percentage points
        
        # Weight by count relative to total
        total_count = summary.get('total_fights', 1)
        weight = count / total_count if total_count > 0 else 0
        weighted_impact = roi_improvement * weight
        
        # Generate fix recommendation
        if 'Split Decision' in spot['subgroup']:
            fix = "Reduce confidence on split decision predictions"
        elif 'underdog' in spot['subgroup'].lower():
            fix = "Adjust expected_score_denom to widen curve OR add underdog boost"
        elif 'Very Long' in spot['subgroup']:
            fix = "More aggressive decay or special layoff handling"
        elif 'drift' in spot.get('issue', '').lower():
            fix = "Add recency weighting or retraining schedule"
        else:
            fix = "Tune parameters for this subgroup"
        
        targets.append({
            'subgroup': f"{spot['category']}: {spot['subgroup']}",
            'current_accuracy': current_acc,
            'target_accuracy': target_acc,
            'current_roi': current_roi,
            'count': count,
            'estimated_impact': weighted_impact,
            'fix': fix
        })
    
    # Sort by estimated impact
    targets.sort(key=lambda x: x['estimated_impact'], reverse=True)
    
    return targets


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(records: pd.DataFrame, all_analyses: Dict, output_dir: str):
    """Create visualization plots for diagnostic results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Baseline Elo Diagnostic Analysis", fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy by Experience Level
    ax1 = axes[0, 0]
    if 'experience' in all_analyses:
        exp_data = all_analyses['experience']['by_bet_on_experience']
        levels = list(EXPERIENCE_LEVELS.keys())
        accs = [exp_data[l]['accuracy'] * 100 if exp_data[l]['accuracy'] is not None else 0 for l in levels]
        colors = ['green' if a >= 55 else 'red' if a < 50 else 'orange' for a in accs]
        ax1.bar(levels, accs, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy by Experience Level')
        ax1.legend()
    
    # Plot 2: Accuracy by Layoff Duration
    ax2 = axes[0, 1]
    if 'layoff' in all_analyses:
        layoff_data = all_analyses['layoff']['by_layoff']
        categories = list(LAYOFF_THRESHOLDS.keys())
        accs = [layoff_data[c]['accuracy'] * 100 if layoff_data[c]['accuracy'] is not None else 0 for c in categories]
        colors = ['green' if a >= 55 else 'red' if a < 50 else 'orange' for a in accs]
        ax2.bar(categories, accs, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy by Layoff Duration')
        ax2.tick_params(axis='x', rotation=15)
    
    # Plot 3: Accuracy by Fight Method
    ax3 = axes[0, 2]
    if 'method' in all_analyses:
        method_data = all_analyses['method']['by_method']
        methods = FIGHT_METHODS
        accs = [method_data[m]['accuracy'] * 100 if method_data.get(m, {}).get('accuracy') is not None else 0 for m in methods]
        colors = ['green' if a >= 55 else 'red' if a < 50 else 'orange' for a in accs]
        ax3.barh(methods, accs, color=colors, alpha=0.7, edgecolor='black')
        ax3.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_title('Accuracy by Fight Method')
    
    # Plot 4: Calibration Curve
    ax4 = axes[1, 0]
    if 'calibration' in all_analyses:
        cal_data = all_analyses['calibration']['by_decile']
        predicted = []
        actual = []
        for bin_name, data in cal_data.items():
            if data['avg_predicted'] is not None and data['actual_rate'] is not None:
                predicted.append(data['avg_predicted'])
                actual.append(data['actual_rate'])
        
        if predicted and actual:
            ax4.scatter(predicted, actual, s=100, alpha=0.7, c='blue', edgecolors='black')
            ax4.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect calibration')
            ax4.set_xlabel('Predicted Probability')
            ax4.set_ylabel('Actual Win Rate')
            ax4.set_title('Calibration Curve')
            ax4.legend()
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
    
    # Plot 5: ROI Over Time
    ax5 = axes[1, 1]
    if 'time' in all_analyses:
        time_data = all_analyses['time']['by_month']
        months = sorted(time_data.keys())
        rois = [time_data[m]['roi'] if time_data[m]['roi'] is not None else 0 for m in months]
        colors = ['green' if r > 0 else 'red' for r in rois]
        ax5.bar(range(len(months)), rois, color=colors, alpha=0.7, edgecolor='black')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax5.set_xticks(range(len(months)))
        ax5.set_xticklabels([m[-5:] for m in months], rotation=45, ha='right')
        ax5.set_ylabel('ROI (%)')
        ax5.set_title('ROI by Month')
    
    # Plot 6: Accuracy by Elo Margin
    ax6 = axes[1, 2]
    if 'elo_margin' in all_analyses:
        margin_data = all_analyses['elo_margin']['by_elo_margin']
        categories = list(ELO_MARGINS.keys())
        accs = [margin_data[c]['accuracy'] * 100 if margin_data.get(c, {}).get('accuracy') else 0 for c in categories]
        colors = ['green' if a >= 55 else 'red' if a < 50 else 'orange' for a in accs]
        ax6.barh(categories, accs, color=colors, alpha=0.7, edgecolor='black')
        ax6.axvline(x=50, color='red', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Accuracy (%)')
        ax6.set_title('Accuracy by Elo Margin')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "baseline_diagnostics.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


# =============================================================================
# Report Generation
# =============================================================================

def print_report(summary: Dict, all_analyses: Dict, weak_spots: List[Dict], 
                 strong_spots: List[Dict], targets: List[Dict]):
    """Print comprehensive diagnostic report."""
    
    print("\n" + "="*80)
    print("BASELINE DIAGNOSTIC REPORT")
    print("="*80)
    
    # Summary
    print(f"\nTotal OOS Fights: {summary['total_fights']}")
    if summary['accuracy']:
        print(f"Accuracy: {summary['accuracy']*100:.1f}%")
    if summary['roi'] is not None:
        print(f"ROI: {summary['roi']:+.2f}%")
    if summary['sharpe'] is not None:
        print(f"Sharpe: {summary['sharpe']:.2f}")
    if summary['win_rate']:
        print(f"Win Rate: {summary['win_rate']*100:.1f}%")
    
    # Weak Spots
    print("\n📊 WEAK SPOTS DETECTED:")
    if weak_spots:
        for spot in weak_spots[:5]:  # Top 5
            acc_str = f"{spot['accuracy']*100:.0f}%" if spot['accuracy'] else "N/A"
            roi_str = f"{spot['roi']:+.1f}%" if spot['roi'] is not None else "N/A"
            print(f"  ⚠️  {spot['category']} - {spot['subgroup']}: "
                  f"Acc={acc_str}, ROI={roi_str}, n={spot['count']}")
    else:
        print("  None detected")
    
    # Strong Spots
    print("\n✅ STRONG SPOTS:")
    if strong_spots:
        for spot in strong_spots[:5]:  # Top 5
            acc_str = f"{spot['accuracy']*100:.0f}%" if spot['accuracy'] else "N/A"
            roi_str = f"{spot['roi']:+.1f}%" if spot['roi'] is not None else "N/A"
            print(f"  ✅ {spot['category']} - {spot['subgroup']}: "
                  f"Acc={acc_str}, ROI={roi_str}")
    else:
        print("  None detected")
    
    # Analysis Details
    analyses = [
        ('experience', 'ANALYSIS 1: By Experience Level', 'by_bet_on_experience'),
        ('layoff', 'ANALYSIS 2: By Layoff Duration', 'by_layoff'),
        ('method', 'ANALYSIS 3: By Fight Method', 'by_method'),
        ('opponent_quality', 'ANALYSIS 4: By Opponent Quality', 'by_opponent_quality'),
        ('elo_margin', 'ANALYSIS 5: By Elo Margin', 'by_elo_margin'),
        ('time', 'ANALYSIS 6: Over Time (Monthly)', 'by_month'),
        ('calibration', 'ANALYSIS 7: Confidence Calibration', 'by_decile'),
    ]
    
    for key, title, subkey in analyses:
        if key not in all_analyses:
            continue
        
        print("\n" + "="*80)
        print(title)
        print("="*80)
        
        data = all_analyses[key].get(subkey, {})
        for subgroup, metrics in data.items():
            if isinstance(metrics, dict) and 'count' in metrics:
                # Handle calibration analysis differently (has different metric structure)
                if key == 'calibration':
                    pred_str = f"{metrics['avg_predicted']*100:.0f}%" if metrics.get('avg_predicted') else "N/A"
                    actual_str = f"{metrics['actual_rate']*100:.0f}%" if metrics.get('actual_rate') else "N/A"
                    status_str = f" ({metrics['status']})" if metrics.get('status') else ""
                    print(f"  {subgroup}: Predicted={pred_str}, Actual={actual_str}, n={metrics['count']}{status_str}")
                else:
                    acc_str = f"{metrics['accuracy']*100:.0f}%" if metrics.get('accuracy') else "N/A"
                    roi_str = f"{metrics['roi']:+.1f}%" if metrics.get('roi') is not None else "N/A"
                    flag_str = f" {metrics['flag']}" if metrics.get('flag') else ""
                    print(f"  {subgroup}: Acc={acc_str}, ROI={roi_str}, n={metrics['count']}{flag_str}")
        
        # Print insights
        insights = all_analyses[key].get('insights', [])
        if insights:
            print("\n🎯 INSIGHT:")
            for insight in insights:
                print(f"   {insight}")
    
    # Optimization Targets
    print("\n" + "="*80)
    print("🎯 TOP OPTIMIZATION TARGETS (by potential impact):")
    print("="*80)
    
    for i, target in enumerate(targets[:5], 1):
        print(f"\n{i}. {target['subgroup']}")
        print(f"   Current: Acc={target['current_accuracy']*100:.0f}%, ROI={target['current_roi']:.1f}%")
        print(f"   Target: Acc={target['target_accuracy']*100:.0f}%")
        print(f"   Estimated impact: +{target['estimated_impact']:.1f}% ROI")
        print(f"   Fix: {target['fix']}")
    
    print("\n" + "="*80)


def save_results_json(output_path: str, summary: Dict, all_analyses: Dict, 
                      weak_spots: List[Dict], strong_spots: List[Dict], 
                      targets: List[Dict]):
    """Save all results to JSON file."""
    
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    results = {
        'summary': convert(summary),
        'analyses': convert(all_analyses),
        'weak_spots': convert(weak_spots),
        'strong_spots': convert(strong_spots),
        'optimization_targets': convert(targets),
        'generated_at': pd.Timestamp.now().isoformat()
    }
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def run_diagnostic_analysis(
    lookback_days: int = 0,
    output_json: Optional[str] = None,
    create_plots: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Run comprehensive diagnostic analysis on baseline Elo predictions.
    
    Args:
        lookback_days: Number of days to analyze (0 = all available data)
        output_json: Path to save JSON results (optional)
        create_plots: Whether to create visualization plots
        verbose: Print detailed output
    
    Returns:
        Dict with all analysis results
    """
    if verbose:
        print("="*80)
        print("BASELINE DIAGNOSTIC ANALYSIS")
        print("="*80)
    
    # Load data
    if verbose:
        print("\nLoading data...")
    df, test_df, odds_df = load_data()
    
    if verbose:
        print(f"Training data: {len(df)} fights")
        print(f"OOS test data: {len(test_df)} fights")
        print(f"Odds data: {len(odds_df)} records")
    
    # Get baseline params
    params = get_baseline_params()
    if verbose:
        print(f"\nUsing baseline params: k={params['k']:.2f}")
    
    # Build prediction records
    if verbose:
        print("\nBuilding prediction records...")
    records = build_prediction_records(df, odds_df, params, lookback_days=lookback_days)
    
    if len(records) == 0:
        print("Error: No valid prediction records generated")
        return {}
    
    if verbose:
        print(f"Generated {len(records)} prediction records")
    
    # Run all analyses
    if verbose:
        print("\nRunning analyses...")
    
    all_analyses = {
        'experience': analyze_by_experience(records),
        'layoff': analyze_by_layoff(records),
        'method': analyze_by_method(records),
        'opponent_quality': analyze_by_opponent_quality(records),
        'elo_margin': analyze_by_elo_margin(records),
        'time': analyze_over_time(records),
        'calibration': analyze_calibration(records),
    }
    
    # Calculate summary
    summary = calculate_summary(records, all_analyses)
    
    # Identify weak and strong spots
    weak_spots = identify_weak_spots(all_analyses)
    strong_spots = identify_strong_spots(all_analyses)
    
    # Calculate optimization targets
    targets = calculate_optimization_targets(weak_spots, summary)
    
    # Print report
    if verbose:
        print_report(summary, all_analyses, weak_spots, strong_spots, targets)
    
    # Create visualizations
    if create_plots:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "images")
        create_visualizations(records, all_analyses, output_dir)
    
    # Save JSON if requested
    if output_json:
        save_results_json(output_json, summary, all_analyses, weak_spots, strong_spots, targets)
    
    return {
        'summary': summary,
        'analyses': all_analyses,
        'weak_spots': weak_spots,
        'strong_spots': strong_spots,
        'optimization_targets': targets,
        'records': records
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic Analysis for Baseline Elo Predictions"
    )
    parser.add_argument(
        "--lookback-days", type=int, default=0,
        help="Number of days to analyze (0 = all available data)"
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Path to save JSON results"
    )
    parser.add_argument(
        "--no-visualizations", action="store_true",
        help="Skip creating visualization plots"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    run_diagnostic_analysis(
        lookback_days=args.lookback_days,
        output_json=args.output_json,
        create_plots=not args.no_visualizations,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
