"""
Comprehensive prediction calibration and consistency metrics for the GA optimization script.

This module provides:
1. Calibration Metrics: ECE, Brier Score, Log Loss, Calibration Slope
2. Consistency Metrics: Variance by opponent tier, method, time period, experience gap
3. Additional Performance Metrics: AUC-ROC, Precision, Recall, Confidence separation
4. Bet Quality Metrics: Kelly Criterion, Drawdown, Sharpe Ratio by period
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from elo_utils import build_fighter_history, has_prior_history


# =========================
# Constants for Fitness Scoring
# =========================

# Default fitness weights for multi-metric optimization
DEFAULT_FITNESS_WEIGHTS = {
    'roi': 0.4,
    'accuracy': 0.2,
    'calibration': 0.15,
    'consistency': 0.15,
    'auc': 0.1
}

# ECE (Expected Calibration Error) thresholds and scaling
# ECE of 0.01 = excellent calibration, 0.1 = poor calibration
DEFAULT_ECE_THRESHOLD = 0.1
ECE_SCALE_FACTOR = 1000  # Scales ECE score to comparable range with ROI

# Consistency variance thresholds and scaling
# Variance of 0.001 = very consistent, 0.01 = inconsistent
DEFAULT_VARIANCE_THRESHOLD = 0.01
VARIANCE_SCALE_FACTOR = 10000  # Scales consistency score to comparable range with ROI


# =========================
# Calibration Metrics
# =========================

def compute_expected_calibration_error(predictions, actuals, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match actual win rates.
    Bins predictions into buckets (0-10%, 10-20%, ..., 90-100%) and computes
    the weighted average of |predicted prob - actual win rate| for each bin.
    
    Args:
        predictions: Array of predicted probabilities (0-1)
        actuals: Array of actual outcomes (0 or 1)
        n_bins: Number of bins for calibration (default 10)
    
    Returns:
        dict: Contains 'ece' (float), 'bin_data' (list of dicts with bin details)
    """
    if len(predictions) == 0:
        return {'ece': None, 'bin_data': []}
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    weighted_error_sum = 0.0
    total_samples = len(predictions)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Get samples in this bin
        if i == n_bins - 1:
            # Include upper bound for last bin
            mask = (predictions >= bin_lower) & (predictions <= bin_upper)
        else:
            mask = (predictions >= bin_lower) & (predictions < bin_upper)
        
        bin_preds = predictions[mask]
        bin_actuals = actuals[mask]
        
        n_in_bin = len(bin_preds)
        if n_in_bin > 0:
            avg_pred = np.mean(bin_preds)
            actual_rate = np.mean(bin_actuals)
            bin_error = abs(avg_pred - actual_rate)
            weighted_error_sum += bin_error * n_in_bin
        else:
            avg_pred = (bin_lower + bin_upper) / 2
            actual_rate = None
            bin_error = None
        
        bin_data.append({
            'bin_lower': bin_lower,
            'bin_upper': bin_upper,
            'count': n_in_bin,
            'avg_predicted': avg_pred if n_in_bin > 0 else None,
            'actual_rate': actual_rate,
            'bin_error': bin_error
        })
    
    ece = weighted_error_sum / total_samples if total_samples > 0 else None
    
    return {
        'ece': ece,
        'bin_data': bin_data
    }


def compute_calibration_slope(predictions, actuals):
    """
    Calculate calibration slope via linear regression.
    
    A slope of 1.0 indicates perfect calibration.
    > 1.0 means underconfident, < 1.0 means overconfident.
    
    Args:
        predictions: Array of predicted probabilities (0-1)
        actuals: Array of actual outcomes (0 or 1)
    
    Returns:
        dict: Contains 'slope', 'intercept', 'is_overconfident', 'is_underconfident'
    """
    if len(predictions) < 2:
        return {
            'slope': None,
            'intercept': None,
            'is_overconfident': None,
            'is_underconfident': None
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Simple linear regression: y = slope * x + intercept
    x_mean = np.mean(predictions)
    y_mean = np.mean(actuals)
    
    numerator = np.sum((predictions - x_mean) * (actuals - y_mean))
    denominator = np.sum((predictions - x_mean) ** 2)
    
    if denominator == 0:
        return {
            'slope': None,
            'intercept': None,
            'is_overconfident': None,
            'is_underconfident': None
        }
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Interpret: slope < 1.0 means overconfident (predictions spread too wide)
    # slope > 1.0 means underconfident (predictions too close to 0.5)
    is_overconfident = slope < 0.9  # Using 0.9 as threshold
    is_underconfident = slope > 1.1  # Using 1.1 as threshold
    
    return {
        'slope': slope,
        'intercept': intercept,
        'is_overconfident': is_overconfident,
        'is_underconfident': is_underconfident
    }


def compute_all_calibration_metrics(predictions, actuals, n_bins=10):
    """
    Compute all calibration metrics at once.
    
    Args:
        predictions: Array of predicted probabilities (0-1)
        actuals: Array of actual outcomes (0 or 1)
        n_bins: Number of bins for ECE calculation
    
    Returns:
        dict: Contains all calibration metrics
    """
    if len(predictions) == 0:
        return {
            'ece': None,
            'brier_score': None,
            'log_loss': None,
            'calibration_slope': None,
            'calibration_intercept': None,
            'is_overconfident': None,
            'is_underconfident': None,
            'bin_data': [],
            'total_predictions': 0
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # ECE
    ece_result = compute_expected_calibration_error(predictions, actuals, n_bins)
    
    # Brier Score
    brier_score = np.mean((predictions - actuals) ** 2)
    
    # Log Loss
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) +
        (1 - actuals) * np.log(1 - predictions_clipped)
    )
    
    # Calibration Slope
    slope_result = compute_calibration_slope(predictions, actuals)
    
    return {
        'ece': ece_result['ece'],
        'brier_score': brier_score,
        'log_loss': log_loss,
        'calibration_slope': slope_result['slope'],
        'calibration_intercept': slope_result['intercept'],
        'is_overconfident': slope_result['is_overconfident'],
        'is_underconfident': slope_result['is_underconfident'],
        'bin_data': ece_result['bin_data'],
        'total_predictions': len(predictions)
    }


# =========================
# Consistency Metrics
# =========================

def compute_consistency_by_elo_tier(df_with_elo, predictions, actuals, elo_diffs, 
                                     tier_boundaries=None):
    """
    Analyze prediction accuracy across different Elo tiers (opponent strength).
    
    Args:
        df_with_elo: DataFrame with Elo ratings
        predictions: Array of predicted probabilities
        actuals: Array of actual outcomes
        elo_diffs: Array of Elo differences (fighter - opponent)
        tier_boundaries: List of Elo diff boundaries for tiers.
                        Default: [-200, -100, 0, 100, 200] creates 6 tiers
    
    Returns:
        dict: Accuracy and ROI metrics for each Elo tier
    """
    if tier_boundaries is None:
        tier_boundaries = [-200, -100, 0, 100, 200]
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    elo_diffs = np.array(elo_diffs)
    
    results = {}
    
    # Create tier labels
    tier_names = []
    boundaries = [-float('inf')] + tier_boundaries + [float('inf')]
    
    for i in range(len(boundaries) - 1):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if lower == -float('inf'):
            name = f"<{upper}"
        elif upper == float('inf'):
            name = f">={lower}"
        else:
            name = f"[{lower}, {upper})"
        tier_names.append(name)
    
    for i, name in enumerate(tier_names):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        
        mask = (elo_diffs >= lower) & (elo_diffs < upper)
        tier_preds = predictions[mask]
        tier_actuals = actuals[mask]
        
        n_in_tier = len(tier_preds)
        if n_in_tier > 0:
            pred_winners = (tier_preds > 0.5).astype(int)
            accuracy = np.mean(pred_winners == tier_actuals)
            avg_pred = np.mean(tier_preds)
            actual_win_rate = np.mean(tier_actuals)
        else:
            accuracy = None
            avg_pred = None
            actual_win_rate = None
        
        results[name] = {
            'count': n_in_tier,
            'accuracy': accuracy,
            'avg_predicted_prob': avg_pred,
            'actual_win_rate': actual_win_rate
        }
    
    # Calculate variance across tiers (consistency metric)
    accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    if len(accuracies) > 1:
        accuracy_variance = np.var(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
    else:
        accuracy_variance = None
        accuracy_range = None
    
    return {
        'tiers': results,
        'accuracy_variance': accuracy_variance,
        'accuracy_range': accuracy_range,
        'high_variance_flag': accuracy_range > 0.05 if accuracy_range else None
    }


def compute_consistency_by_method(df_with_elo, method_mapping=None):
    """
    Analyze prediction accuracy by fight method (KO, Submission, Decision).
    
    Args:
        df_with_elo: DataFrame with Elo ratings and method columns
        method_mapping: Custom mapping of method column names to method types
    
    Returns:
        dict: Accuracy metrics for each fight method
    """
    # Default method column mapping
    if method_mapping is None:
        method_mapping = {
            'ko': 'KO/TKO',
            'kod': 'KO/TKO',
            'subw': 'Submission',
            'subwd': 'Submission',
            'udec': 'Unanimous Decision',
            'udecd': 'Unanimous Decision',
            'sdec': 'Split Decision',
            'sdecd': 'Split Decision',
            'mdec': 'Majority Decision',
            'mdecd': 'Majority Decision'
        }
    
    results = {}
    
    # Determine method for each fight
    methods = []
    for _, row in df_with_elo.iterrows():
        method_found = None
        for col, method_name in method_mapping.items():
            if col in row and (row[col] == 1 or str(row.get(col, '')).strip() == '1'):
                method_found = method_name
                break
        methods.append(method_found or 'Unknown')
    
    df_copy = df_with_elo.copy()
    df_copy['_method'] = methods
    
    # Build fighter history for prior fight check
    hist = build_fighter_history(df_copy)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    # Calculate metrics per method
    for method_name in set(methods):
        method_df = df_copy[df_copy['_method'] == method_name]
        
        predictions = []
        actuals = []
        
        for _, row in method_df.iterrows():
            result = row.get("result")
            if result not in (0, 1):
                continue
            if pd.isna(row.get("DATE")):
                continue
            if row.get("precomp_elo") == row.get("opp_precomp_elo"):
                continue
            
            fighter = row.get("FIGHTER")
            opponent = row.get("opp_FIGHTER")
            
            # Skip if no prior history
            if not has_prior_history(first_dates, fighter, row["DATE"]):
                continue
            if not has_prior_history(first_dates, opponent, row["DATE"]):
                continue
            
            elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
            pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            
            predictions.append(pred_prob)
            actuals.append(int(result))
        
        n_fights = len(predictions)
        if n_fights > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            pred_winners = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_winners == actuals)
            brier = np.mean((predictions - actuals) ** 2)
        else:
            accuracy = None
            brier = None
        
        results[method_name] = {
            'count': n_fights,
            'accuracy': accuracy,
            'brier_score': brier
        }
    
    # Calculate variance across methods
    accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    if len(accuracies) > 1:
        accuracy_variance = np.var(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
    else:
        accuracy_variance = None
        accuracy_range = None
    
    return {
        'methods': results,
        'accuracy_variance': accuracy_variance,
        'accuracy_range': accuracy_range,
        'high_variance_flag': accuracy_range > 0.05 if accuracy_range else None
    }


def compute_consistency_by_time_period(df_with_elo, period='M'):
    """
    Analyze prediction accuracy over time periods to detect drift.
    
    Args:
        df_with_elo: DataFrame with Elo ratings and DATE column
        period: Pandas frequency string ('M' for monthly, 'Q' for quarterly)
    
    Returns:
        dict: Accuracy metrics for each time period
    """
    df_copy = df_with_elo.copy()
    df_copy['_period'] = pd.to_datetime(df_copy['DATE']).dt.to_period(period)
    
    # Build fighter history
    hist = build_fighter_history(df_copy)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    results = {}
    
    for period_val, period_df in df_copy.groupby('_period'):
        predictions = []
        actuals = []
        
        for _, row in period_df.iterrows():
            result = row.get("result")
            if result not in (0, 1):
                continue
            if pd.isna(row.get("DATE")):
                continue
            if row.get("precomp_elo") == row.get("opp_precomp_elo"):
                continue
            
            fighter = row.get("FIGHTER")
            opponent = row.get("opp_FIGHTER")
            
            if not has_prior_history(first_dates, fighter, row["DATE"]):
                continue
            if not has_prior_history(first_dates, opponent, row["DATE"]):
                continue
            
            elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
            pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            
            predictions.append(pred_prob)
            actuals.append(int(result))
        
        n_fights = len(predictions)
        if n_fights > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            pred_winners = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_winners == actuals)
        else:
            accuracy = None
        
        results[str(period_val)] = {
            'count': n_fights,
            'accuracy': accuracy
        }
    
    # Calculate trend (regression slope over time)
    periods_with_accuracy = [(k, v['accuracy']) 
                             for k, v in sorted(results.items()) 
                             if v['accuracy'] is not None]
    
    if len(periods_with_accuracy) >= 3:
        x = np.arange(len(periods_with_accuracy))
        y = np.array([accuracy for _, accuracy in periods_with_accuracy])
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        trend = numerator / denominator if denominator > 0 else 0
    else:
        trend = None
    
    # Calculate variance
    accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    if len(accuracies) > 1:
        accuracy_variance = np.var(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
    else:
        accuracy_variance = None
        accuracy_range = None
    
    return {
        'periods': results,
        'trend': trend,  # Positive = improving, Negative = degrading
        'accuracy_variance': accuracy_variance,
        'accuracy_range': accuracy_range,
        'is_degrading': trend < -0.005 if trend is not None else None
    }


def compute_consistency_by_experience_gap(df_with_elo):
    """
    Analyze prediction accuracy by experience gap between fighters.
    
    Uses precomp_boutcount columns to determine experience.
    
    Args:
        df_with_elo: DataFrame with Elo ratings and bout count columns
    
    Returns:
        dict: Accuracy metrics for different experience gap scenarios
    """
    df_copy = df_with_elo.copy()
    
    # Ensure boutcount columns exist
    if 'precomp_boutcount' not in df_copy.columns:
        return {
            'gaps': {},
            'accuracy_variance': None,
            'accuracy_range': None,
            'high_variance_flag': None
        }
    
    # Build fighter history
    hist = build_fighter_history(df_copy)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    # Define experience gap buckets
    gap_buckets = {
        'balanced (0-3)': (0, 3),
        'slight gap (4-10)': (4, 10),
        'moderate gap (11-20)': (11, 20),
        'large gap (>20)': (21, float('inf'))
    }
    
    results = {}
    
    for bucket_name, (min_gap, max_gap) in gap_buckets.items():
        predictions = []
        actuals = []
        
        for _, row in df_copy.iterrows():
            result = row.get("result")
            if result not in (0, 1):
                continue
            if pd.isna(row.get("DATE")):
                continue
            if row.get("precomp_elo") == row.get("opp_precomp_elo"):
                continue
            
            fighter = row.get("FIGHTER")
            opponent = row.get("opp_FIGHTER")
            
            if not has_prior_history(first_dates, fighter, row["DATE"]):
                continue
            if not has_prior_history(first_dates, opponent, row["DATE"]):
                continue
            
            # Calculate experience gap
            f_bouts = row.get("precomp_boutcount", 0) or 0
            opp_bouts = row.get("opp_precomp_boutcount", 0) or 0
            exp_gap = abs(f_bouts - opp_bouts)
            
            if min_gap <= exp_gap <= max_gap:
                elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
                pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
                
                predictions.append(pred_prob)
                actuals.append(int(result))
        
        n_fights = len(predictions)
        if n_fights > 0:
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            pred_winners = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_winners == actuals)
        else:
            accuracy = None
        
        results[bucket_name] = {
            'count': n_fights,
            'accuracy': accuracy
        }
    
    # Calculate variance
    accuracies = [r['accuracy'] for r in results.values() if r['accuracy'] is not None]
    if len(accuracies) > 1:
        accuracy_variance = np.var(accuracies)
        accuracy_range = max(accuracies) - min(accuracies)
    else:
        accuracy_variance = None
        accuracy_range = None
    
    return {
        'gaps': results,
        'accuracy_variance': accuracy_variance,
        'accuracy_range': accuracy_range,
        'high_variance_flag': accuracy_range > 0.05 if accuracy_range else None
    }


def compute_all_consistency_metrics(df_with_elo):
    """
    Compute all consistency metrics at once.
    
    Args:
        df_with_elo: DataFrame with Elo ratings and relevant columns
    
    Returns:
        dict: All consistency metrics grouped by analysis type
    """
    # First, extract predictions and actuals for tier analysis
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    predictions = []
    actuals = []
    elo_diffs = []
    
    for _, row in df_with_elo.iterrows():
        result = row.get("result")
        if result not in (0, 1):
            continue
        if pd.isna(row.get("DATE")):
            continue
        if row.get("precomp_elo") == row.get("opp_precomp_elo"):
            continue
        
        fighter = row.get("FIGHTER")
        opponent = row.get("opp_FIGHTER")
        
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        predictions.append(pred_prob)
        actuals.append(int(result))
        elo_diffs.append(elo_diff)
    
    by_tier = compute_consistency_by_elo_tier(df_with_elo, predictions, actuals, elo_diffs)
    by_method = compute_consistency_by_method(df_with_elo)
    by_time = compute_consistency_by_time_period(df_with_elo, period='M')
    by_experience = compute_consistency_by_experience_gap(df_with_elo)
    
    # Calculate overall consistency score (lower is better)
    variances = []
    if by_tier['accuracy_variance'] is not None:
        variances.append(by_tier['accuracy_variance'])
    if by_method['accuracy_variance'] is not None:
        variances.append(by_method['accuracy_variance'])
    if by_time['accuracy_variance'] is not None:
        variances.append(by_time['accuracy_variance'])
    if by_experience['accuracy_variance'] is not None:
        variances.append(by_experience['accuracy_variance'])
    
    overall_variance = np.mean(variances) if variances else None
    
    # Flag any high-variance dimensions
    flags = []
    if by_tier.get('high_variance_flag'):
        flags.append('elo_tier')
    if by_method.get('high_variance_flag'):
        flags.append('method')
    if by_experience.get('high_variance_flag'):
        flags.append('experience')
    if by_time.get('is_degrading'):
        flags.append('time_degrading')
    
    return {
        'by_elo_tier': by_tier,
        'by_method': by_method,
        'by_time_period': by_time,
        'by_experience_gap': by_experience,
        'overall_variance': overall_variance,
        'high_variance_flags': flags
    }


# =========================
# Additional Performance Metrics
# =========================

def compute_auc_roc(predictions, actuals):
    """
    Calculate Area Under ROC Curve.
    
    Args:
        predictions: Array of predicted probabilities
        actuals: Array of actual outcomes
    
    Returns:
        float: AUC-ROC score (0.5 = random, 1.0 = perfect)
    """
    if len(predictions) < 2:
        return None
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Need both classes present
    if len(np.unique(actuals)) < 2:
        return None
    
    try:
        return roc_auc_score(actuals, predictions)
    except Exception:
        return None


def compute_precision_recall(predictions, actuals, threshold=0.5):
    """
    Calculate Precision and Recall for favorites and underdogs.
    
    Args:
        predictions: Array of predicted probabilities
        actuals: Array of actual outcomes
        threshold: Probability threshold for classification
    
    Returns:
        dict: Contains precision, recall for favorites (>threshold) and underdogs
    """
    if len(predictions) < 2:
        return {
            'favorites': {'precision': None, 'recall': None, 'count': 0},
            'underdogs': {'precision': None, 'recall': None, 'count': 0}
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Favorites: predictions > threshold, betting on the predicted winner
    # For favorites: we predict win (1) when prob > threshold
    pred_labels = (predictions > threshold).astype(int)
    
    # Calculate overall precision/recall
    if len(np.unique(pred_labels)) < 2 or len(np.unique(actuals)) < 2:
        return {
            'favorites': {'precision': None, 'recall': None, 'count': 0},
            'underdogs': {'precision': None, 'recall': None, 'count': 0}
        }
    
    try:
        favorites_mask = predictions > threshold
        underdogs_mask = predictions <= threshold
        
        # Favorites: we predicted win
        fav_preds = pred_labels[favorites_mask]
        fav_actuals = actuals[favorites_mask]
        
        # Underdogs: we predicted loss (but need to check if upset occurred)
        und_preds = pred_labels[underdogs_mask]
        und_actuals = actuals[underdogs_mask]
        
        # Precision for favorites: of fights we predicted to win, how many won
        fav_precision = np.mean(fav_actuals == 1) if len(fav_actuals) > 0 else None
        # Recall for favorites: of actual wins, how many did we predict
        actual_wins = actuals == 1
        predicted_wins = pred_labels == 1
        fav_recall = (np.sum(actual_wins & predicted_wins) / np.sum(actual_wins) 
                      if np.sum(actual_wins) > 0 else None)
        
        return {
            'favorites': {
                'precision': fav_precision,
                'recall': fav_recall,
                'count': len(fav_actuals)
            },
            'underdogs': {
                'precision': np.mean(und_actuals == 0) if len(und_actuals) > 0 else None,
                'recall': None,  # Underdog recall is less meaningful
                'count': len(und_actuals)
            }
        }
    except Exception:
        return {
            'favorites': {'precision': None, 'recall': None, 'count': 0},
            'underdogs': {'precision': None, 'recall': None, 'count': 0}
        }


def compute_confidence_separation(predictions, actuals):
    """
    Calculate confidence score separation between winners and losers.
    
    Higher separation indicates the model gives more confident predictions
    to eventual winners vs losers.
    
    Args:
        predictions: Array of predicted probabilities
        actuals: Array of actual outcomes
    
    Returns:
        dict: Average confidence for winners, losers, and separation score
    """
    if len(predictions) == 0:
        return {
            'avg_confidence_winners': None,
            'avg_confidence_losers': None,
            'separation': None
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    winner_mask = actuals == 1
    loser_mask = actuals == 0
    
    avg_conf_winners = np.mean(predictions[winner_mask]) if np.sum(winner_mask) > 0 else None
    avg_conf_losers = np.mean(predictions[loser_mask]) if np.sum(loser_mask) > 0 else None
    
    if avg_conf_winners is not None and avg_conf_losers is not None:
        separation = avg_conf_winners - avg_conf_losers
    else:
        separation = None
    
    return {
        'avg_confidence_winners': avg_conf_winners,
        'avg_confidence_losers': avg_conf_losers,
        'separation': separation
    }


def compute_roi_by_confidence_decile(bet_records, predictions):
    """
    Calculate ROI for each confidence decile.
    
    Args:
        bet_records: List of bet record dicts with 'profit', 'bet_amount', 'bet_won'
        predictions: Array of predicted probabilities corresponding to bets
    
    Returns:
        dict: ROI and metrics for each decile (0-10%, 10-20%, ..., 90-100%)
    """
    if len(bet_records) == 0 or len(predictions) == 0:
        return {'deciles': {}, 'best_decile': None, 'worst_decile': None}
    
    predictions = np.array(predictions)
    
    # Create DataFrame for easier grouping
    df = pd.DataFrame(bet_records)
    df['prediction'] = predictions
    
    # Assign to deciles
    df['decile'] = pd.cut(df['prediction'], 
                         bins=np.linspace(0, 1, 11),
                         labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)],
                         include_lowest=True)
    
    results = {}
    for decile, group in df.groupby('decile', observed=True):
        if len(group) > 0:
            total_wagered = group['bet_amount'].sum()
            total_profit = group['profit'].sum()
            roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
            win_rate = group['bet_won'].mean()
        else:
            roi = None
            win_rate = None
            total_wagered = 0
        
        results[str(decile)] = {
            'count': len(group),
            'roi': roi,
            'win_rate': win_rate,
            'total_wagered': total_wagered
        }
    
    # Find best and worst deciles
    roi_values = [(k, v['roi']) for k, v in results.items() if v['roi'] is not None]
    if roi_values:
        best_decile = max(roi_values, key=lambda x: x[1])[0]
        worst_decile = min(roi_values, key=lambda x: x[1])[0]
    else:
        best_decile = None
        worst_decile = None
    
    return {
        'deciles': results,
        'best_decile': best_decile,
        'worst_decile': worst_decile
    }


def compute_all_performance_metrics(predictions, actuals, bet_records=None):
    """
    Compute all additional performance metrics at once.
    
    Args:
        predictions: Array of predicted probabilities
        actuals: Array of actual outcomes
        bet_records: Optional list of bet records for ROI by decile
    
    Returns:
        dict: All performance metrics
    """
    auc_roc = compute_auc_roc(predictions, actuals)
    prec_recall = compute_precision_recall(predictions, actuals)
    conf_sep = compute_confidence_separation(predictions, actuals)
    
    roi_decile = {'deciles': {}, 'best_decile': None, 'worst_decile': None}
    if bet_records is not None and len(bet_records) == len(predictions):
        roi_decile = compute_roi_by_confidence_decile(bet_records, predictions)
    
    return {
        'auc_roc': auc_roc,
        'precision_recall': prec_recall,
        'confidence_separation': conf_sep,
        'roi_by_decile': roi_decile
    }


# =========================
# Bet Quality Metrics
# =========================

def compute_kelly_criterion(pred_prob, decimal_odds):
    """
    Calculate Kelly Criterion bet sizing recommendation.
    
    Kelly Criterion: f* = (bp - q) / b
    where:
    - f* = fraction of bankroll to bet
    - b = decimal odds - 1 (net odds)
    - p = probability of winning
    - q = probability of losing = 1 - p
    
    Args:
        pred_prob: Predicted probability of winning (0-1)
        decimal_odds: Decimal odds for the bet
    
    Returns:
        float: Recommended fraction of bankroll to bet (can be negative = don't bet)
    """
    if pred_prob is None or decimal_odds is None or decimal_odds <= 1:
        return None
    
    b = decimal_odds - 1  # Net odds
    p = pred_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return kelly


def compute_max_drawdown(cumulative_profits):
    """
    Calculate maximum drawdown from a series of cumulative profits.
    
    Args:
        cumulative_profits: Array or list of cumulative profit values
    
    Returns:
        dict: Contains 'max_drawdown' (absolute), 'max_drawdown_pct' (percentage)
    """
    if len(cumulative_profits) == 0:
        return {'max_drawdown': None, 'max_drawdown_pct': None, 'drawdown_duration': None}
    
    cumulative = np.array(cumulative_profits)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Calculate drawdown at each point
    drawdowns = running_max - cumulative
    
    max_drawdown = np.max(drawdowns)
    
    # Calculate percentage drawdown (relative to peak)
    peak_at_max_dd = running_max[np.argmax(drawdowns)]
    if peak_at_max_dd > 0:
        max_drawdown_pct = (max_drawdown / peak_at_max_dd) * 100
    else:
        max_drawdown_pct = None
    
    # Calculate drawdown duration (periods from peak to recovery)
    if max_drawdown > 0:
        in_drawdown = drawdowns > 0
        if np.any(in_drawdown):
            # Count consecutive periods in drawdown
            changes = np.diff(np.concatenate(([0], in_drawdown.astype(int), [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > 0 and len(ends) > 0:
                durations = ends - starts
                drawdown_duration = np.max(durations)
            else:
                drawdown_duration = None
        else:
            drawdown_duration = 0
    else:
        drawdown_duration = 0
    
    return {
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'drawdown_duration': drawdown_duration
    }


def compute_sharpe_ratio_by_period(bet_records, period='M'):
    """
    Calculate Sharpe ratio for each time period.
    
    Args:
        bet_records: List of bet record dicts with 'date', 'profit', 'bet_amount'
        period: Pandas frequency string ('M' for monthly, 'Q' for quarterly)
    
    Returns:
        dict: Sharpe ratio and metrics for each period
    """
    if len(bet_records) == 0:
        return {'periods': {}, 'rolling_sharpe': None, 'sharpe_stability': None}
    
    df = pd.DataFrame(bet_records)
    df['date'] = pd.to_datetime(df['date'])
    df['period'] = df['date'].dt.to_period(period)
    
    results = {}
    sharpe_values = []
    
    for period_val, group in df.groupby('period'):
        total_wagered = group['bet_amount'].sum()
        total_profit = group['profit'].sum()
        
        if total_wagered > 0:
            roi = (total_profit / total_wagered) * 100
            
            # Calculate daily ROIs within period
            daily = group.groupby(group['date'].dt.date).agg({
                'profit': 'sum',
                'bet_amount': 'sum'
            })
            
            if len(daily) >= 2:
                daily['roi'] = (daily['profit'] / daily['bet_amount']) * 100
                mean_roi = daily['roi'].mean()
                std_roi = daily['roi'].std()
                
                if std_roi > 0:
                    sharpe = mean_roi / std_roi
                    sharpe_values.append(sharpe)
                else:
                    sharpe = None
            else:
                sharpe = None
        else:
            roi = 0
            sharpe = None
        
        results[str(period_val)] = {
            'roi': roi,
            'sharpe': sharpe,
            'num_bets': len(group)
        }
    
    # Calculate stability of Sharpe ratio
    if len(sharpe_values) >= 2:
        sharpe_stability = np.std(sharpe_values)
    else:
        sharpe_stability = None
    
    return {
        'periods': results,
        'rolling_sharpe': sharpe_values if sharpe_values else None,
        'sharpe_stability': sharpe_stability
    }


def compute_value_bet_analysis(predictions, implied_probs, actuals, bet_results=None):
    """
    Analyze value bets where model probability exceeds implied odds probability.
    
    Args:
        predictions: Array of predicted probabilities
        implied_probs: Array of implied probabilities from odds
        actuals: Array of actual outcomes
        bet_results: Optional array of bet won/lost
    
    Returns:
        dict: Value bet statistics
    """
    if len(predictions) == 0:
        return {
            'value_bet_count': 0,
            'value_bet_accuracy': None,
            'avg_edge': None,
            'total_edge': 0
        }
    
    predictions = np.array(predictions)
    implied_probs = np.array(implied_probs)
    actuals = np.array(actuals)
    
    # Value bet: predicted prob > implied prob (we have edge)
    value_bet_mask = predictions > implied_probs
    
    value_bet_count = np.sum(value_bet_mask)
    
    if value_bet_count > 0:
        value_actuals = actuals[value_bet_mask]
        value_preds = predictions[value_bet_mask]
        
        # Accuracy on value bets
        pred_winners = (value_preds > 0.5).astype(int)
        value_bet_accuracy = np.mean(pred_winners == value_actuals)
        
        # Edge calculation
        edges = predictions[value_bet_mask] - implied_probs[value_bet_mask]
        avg_edge = np.mean(edges)
        total_edge = np.sum(edges)
    else:
        value_bet_accuracy = None
        avg_edge = None
        total_edge = 0
    
    return {
        'value_bet_count': int(value_bet_count),
        'value_bet_accuracy': value_bet_accuracy,
        'avg_edge': avg_edge,
        'total_edge': total_edge,
        'value_bet_ratio': value_bet_count / len(predictions) if len(predictions) > 0 else 0
    }


def compute_all_bet_quality_metrics(bet_records, predictions=None, implied_probs=None, actuals=None):
    """
    Compute all bet quality metrics at once.
    
    Args:
        bet_records: List of bet record dicts
        predictions: Optional array of predicted probabilities
        implied_probs: Optional array of implied probabilities from odds
        actuals: Optional array of actual outcomes
    
    Returns:
        dict: All bet quality metrics
    """
    results = {}
    
    # Max drawdown
    if bet_records:
        df = pd.DataFrame(bet_records)
        if 'profit' in df.columns:
            cumulative = df['profit'].cumsum().values
            results['drawdown'] = compute_max_drawdown(cumulative)
        else:
            results['drawdown'] = {'max_drawdown': None, 'max_drawdown_pct': None}
        
        # Sharpe by period
        if 'date' in df.columns and 'profit' in df.columns and 'bet_amount' in df.columns:
            results['sharpe_by_period'] = compute_sharpe_ratio_by_period(bet_records)
        else:
            results['sharpe_by_period'] = {'periods': {}, 'rolling_sharpe': None}
    else:
        results['drawdown'] = {'max_drawdown': None, 'max_drawdown_pct': None}
        results['sharpe_by_period'] = {'periods': {}, 'rolling_sharpe': None}
    
    # Value bet analysis
    if predictions is not None and implied_probs is not None and actuals is not None:
        results['value_bets'] = compute_value_bet_analysis(predictions, implied_probs, actuals)
    else:
        results['value_bets'] = {
            'value_bet_count': 0,
            'value_bet_accuracy': None,
            'avg_edge': None
        }
    
    # Kelly criterion summary (average recommended stake)
    if bet_records and 'prediction' in pd.DataFrame(bet_records).columns:
        df = pd.DataFrame(bet_records)
        kelly_values = []
        for _, row in df.iterrows():
            if 'prediction' in row and 'decimal_odds' in row:
                kelly = compute_kelly_criterion(row['prediction'], row['decimal_odds'])
                if kelly is not None:
                    kelly_values.append(kelly)
        
        if kelly_values:
            results['kelly_summary'] = {
                'avg_kelly': np.mean(kelly_values),
                'positive_kelly_pct': np.mean([k > 0 for k in kelly_values]) * 100,
                'max_kelly': max(kelly_values),
                'min_kelly': min(kelly_values)
            }
        else:
            results['kelly_summary'] = {
                'avg_kelly': None,
                'positive_kelly_pct': None
            }
    else:
        results['kelly_summary'] = {'avg_kelly': None, 'positive_kelly_pct': None}
    
    return results


# =========================
# Master Function for All Metrics
# =========================

def compute_comprehensive_metrics(df_with_elo, bet_records=None, odds_df=None):
    """
    Compute all prediction calibration and consistency metrics.
    
    This is the main entry point for computing all metrics for the GA fitness function.
    
    Args:
        df_with_elo: DataFrame with Elo ratings already calculated
        bet_records: Optional list of bet record dicts for betting metrics
        odds_df: Optional odds DataFrame for value bet analysis
    
    Returns:
        dict: Comprehensive metrics including:
            - calibration: ECE, Brier, LogLoss, Calibration Slope
            - consistency: By tier, method, time, experience
            - performance: AUC-ROC, Precision, Recall, Confidence separation
            - betting: Drawdown, Sharpe by period, Kelly, Value bets
    """
    # Extract predictions and actuals
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    predictions = []
    actuals = []
    elo_diffs = []
    implied_probs = []
    
    for _, row in df_with_elo.iterrows():
        result = row.get("result")
        if result not in (0, 1):
            continue
        if pd.isna(row.get("DATE")):
            continue
        if row.get("precomp_elo") == row.get("opp_precomp_elo"):
            continue
        
        fighter = row.get("FIGHTER")
        opponent = row.get("opp_FIGHTER")
        
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        predictions.append(pred_prob)
        actuals.append(int(result))
        elo_diffs.append(elo_diff)
        
        # Calculate implied probability from odds if available
        if odds_df is not None:
            # Try to find odds for this fight
            avg_odds = row.get("avg_odds")
            if pd.notna(avg_odds):
                # Convert American odds to implied probability
                if avg_odds > 0:
                    implied = 100 / (avg_odds + 100)
                else:
                    implied = abs(avg_odds) / (abs(avg_odds) + 100)
                implied_probs.append(implied)
            else:
                implied_probs.append(0.5)  # Default to 50%
    
    predictions = np.array(predictions) if predictions else np.array([])
    actuals = np.array(actuals) if actuals else np.array([])
    elo_diffs = np.array(elo_diffs) if elo_diffs else np.array([])
    implied_probs = np.array(implied_probs) if implied_probs else None
    
    # Compute all metric categories
    calibration = compute_all_calibration_metrics(predictions, actuals)
    consistency = compute_all_consistency_metrics(df_with_elo)
    performance = compute_all_performance_metrics(predictions, actuals, bet_records)
    betting = compute_all_bet_quality_metrics(bet_records, predictions, implied_probs, actuals)
    
    return {
        'calibration': calibration,
        'consistency': consistency,
        'performance': performance,
        'betting': betting,
        'summary': {
            'total_predictions': len(predictions),
            'overall_accuracy': np.mean((predictions > 0.5).astype(int) == actuals) if len(predictions) > 0 else None,
            'ece': calibration['ece'],
            'brier_score': calibration['brier_score'],
            'auc_roc': performance['auc_roc'],
            'calibration_slope': calibration['calibration_slope'],
            'consistency_variance': consistency['overall_variance'],
            'high_variance_flags': consistency['high_variance_flags']
        }
    }


def compute_composite_fitness(metrics, weights=None):
    """
    Compute a composite fitness score from multiple metrics.
    
    This allows the GA to optimize on a weighted combination of metrics
    rather than just ROI alone.
    
    Args:
        metrics: Dict from compute_comprehensive_metrics or similar
        weights: Dict with weights for each metric component:
            - 'roi': Weight for ROI (default 0.4)
            - 'accuracy': Weight for accuracy (default 0.2)
            - 'calibration': Weight for calibration (lower ECE is better) (default 0.15)
            - 'consistency': Weight for consistency (lower variance is better) (default 0.15)
            - 'auc': Weight for AUC-ROC (default 0.1)
    
    Returns:
        float: Composite fitness score (higher is better)
    """
    if weights is None:
        weights = DEFAULT_FITNESS_WEIGHTS.copy()
    else:
        # Merge with defaults for any missing keys
        merged = DEFAULT_FITNESS_WEIGHTS.copy()
        merged.update(weights)
        weights = merged
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    fitness = 0.0
    
    # ROI component (already from extended metrics, not in this dict)
    roi = metrics.get('roi_percent', 0) or 0
    fitness += weights.get('roi', 0) * roi
    
    # Accuracy component (scale to similar magnitude as ROI)
    accuracy = metrics.get('summary', {}).get('overall_accuracy', 0) or 0
    fitness += weights.get('accuracy', 0) * (accuracy * 100)  # Convert to percentage
    
    # Calibration component (lower ECE is better, so invert and scale)
    ece = metrics.get('summary', {}).get('ece', DEFAULT_ECE_THRESHOLD) or DEFAULT_ECE_THRESHOLD
    # Scale: ECE of 0.01 = 99, ECE of 0.1 = 0, ECE of 0.05 = 50
    calibration_score = max(0, (DEFAULT_ECE_THRESHOLD - ece) * ECE_SCALE_FACTOR)
    fitness += weights.get('calibration', 0) * calibration_score
    
    # Consistency component (lower variance is better)
    variance = metrics.get('summary', {}).get('consistency_variance', DEFAULT_VARIANCE_THRESHOLD) or DEFAULT_VARIANCE_THRESHOLD
    # Scale: variance of 0.001 = 90, variance of 0.01 = 0
    consistency_score = max(0, (DEFAULT_VARIANCE_THRESHOLD - variance) * VARIANCE_SCALE_FACTOR)
    fitness += weights.get('consistency', 0) * consistency_score
    
    # AUC-ROC component (scale to similar magnitude)
    auc = metrics.get('summary', {}).get('auc_roc', 0.5) or 0.5
    # Scale: AUC of 0.5 = 0, AUC of 1.0 = 100
    auc_score = (auc - 0.5) * 200
    fitness += weights.get('auc', 0) * auc_score
    
    return fitness
