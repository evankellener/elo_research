import random
import math
import json
import pandas as pd
import numpy as np
import unicodedata
from multiprocessing import Pool, cpu_count
from functools import partial
from elo_utils import (
    mov_factor, build_fighter_history, has_prior_history, add_bout_counts, apply_decay,
    apply_multiphase_decay, build_fighter_weight_history, detect_weight_change, calculate_expected_value
)
from prediction_metrics import (
    compute_comprehensive_metrics, compute_composite_fitness,
    compute_all_calibration_metrics, compute_all_consistency_metrics,
    compute_all_performance_metrics, compute_all_bet_quality_metrics
)


def normalize_name(name):
    """
    Normalize fighter name for matching (handles special characters, middle names, etc.)
    Converts special characters like 'ę' to 'e' by normalizing unicode and removing combining marks.
    """
    if pd.isna(name):
        return ""
    # Normalize unicode (decomposes characters like 'ę' into 'e' + combining mark)
    name = unicodedata.normalize('NFKD', str(name))
    # Filter out combining characters (like the combining mark after decomposed 'ę')
    name = ''.join(c for c in name if not unicodedata.combining(c))
    # Convert to lowercase and remove extra spaces
    name = ' '.join(name.lower().split())
    return name


def find_fighter_match(test_name, training_names):
    """
    Find the best matching fighter name from training data.
    Handles cases like:
    - "Jose Miguel Delgado" vs "Jose Delgado"
    - "Mateusz Rębecki" vs "Mateusz Rebecki"
    """
    test_normalized = normalize_name(test_name)
    
    # First try exact match
    for train_name in training_names:
        if normalize_name(train_name) == test_normalized:
            return train_name
    
    # Try matching by last name and first name (handles middle name variations)
    test_parts = test_normalized.split()
    if len(test_parts) >= 2:
        test_first = test_parts[0]
        test_last = test_parts[-1]
        for train_name in training_names:
            train_normalized = normalize_name(train_name)
            train_parts = train_normalized.split()
            if len(train_parts) >= 2:
                train_first = train_parts[0]
                train_last = train_parts[-1]
                # Match if first and last names match (ignoring middle names)
                if test_first == train_first and test_last == train_last:
                    return train_name
    
    return None


# =========================
# Elo with Method of Victory
# =========================


def run_basic_elo(df, k=32, base_elo=1500, denominator=400, mov_params=None, draw_k_factor=0.5,
                  decay_mode="none", decay_rate=0.0, min_days=180,
                  multiphase_decay_params=None, weight_adjust_params=None, weight_history=None):
    """
    Core Elo engine, with optional method of victory multipliers and decay.
    
    Args:
        df: DataFrame with fight data
        k: Base K-factor (default 32)
        base_elo: Starting Elo rating (default 1500)
        denominator: Elo denominator (default 400) - can be optimized with --optimize-elo-denom
        mov_params: Dict with MoV weights, or None for no adjustment
        draw_k_factor: Multiplier for K-factor in draws (default 0.5)
        decay_mode: "linear", "exponential", or "none" (default "none")
        decay_rate: Decay rate per day (default 0.0)
        min_days: Minimum days before decay starts (default 180)
        multiphase_decay_params: Dict with multiphase decay parameters or None:
            - quick_succession_days: Days threshold for quick succession bump
            - quick_succession_bump: Multiplier for recent fighters (> 1.0 = boost)
            - decay_days: Days threshold for decay to start
            - multiphase_decay_rate: Exponential decay rate
        weight_adjust_params: Dict with weight adjustment parameters or None:
            - weight_up_precomp_penalty: Penalty for moving up in weight (< 1.0)
            - weight_up_postcomp_bonus: Bonus for winning in higher weight class
        weight_history: Pre-built fighter weight history (from build_fighter_weight_history)
    
    Note: Assumes df is sorted by DATE. If not, results may be incorrect.
    
    The decay is applied to precomp_elo BEFORE prediction but does NOT affect 
    the stored postcomp_elo. The adjusted precomp_elo is used only for 
    calculating expected scores and making predictions.
    """
    df = df.copy()
    # Ensure dataframe is sorted by date for correct chronological processing
    if "DATE" in df.columns:
        df = df.sort_values("DATE").reset_index(drop=True)
    ratings = {}
    last_fight_dates = {}  # Track last fight date for each fighter
    pre, post, opp_pre, opp_post = [], [], [], []
    adjusted_pre, adjusted_opp_pre = [], []  # Track decay-adjusted precomp_elo
    
    # Check which features are enabled
    use_multiphase_decay = multiphase_decay_params is not None
    use_weight_adjust = weight_adjust_params is not None and weight_history is not None

    for idx, row in df.iterrows():
        f1, f2, res = row["FIGHTER"], row["opp_FIGHTER"], row["result"]
        current_date = row.get("DATE")
        
        # Check if this is a draw (both win and loss are 0)
        # Note: Default value of 1 means if columns don't exist, it won't be detected as a draw
        # This assumes "win" and "loss" columns always exist in the data
        is_draw = (row.get("win", 1) == 0) and (row.get("loss", 1) == 0)
        
        # For draws, use 0.5 as the result (half win for each fighter)
        if is_draw:
            res = 0.5
        
        r1 = ratings.get(f1, base_elo)
        r2 = ratings.get(f2, base_elo)
        
        # Apply decay adjustment for predictions (if decay is enabled)
        r1_adjusted = r1
        r2_adjusted = r2
        
        # Calculate days since last fight for each fighter
        last_date_f1 = last_fight_dates.get(f1)
        last_date_f2 = last_fight_dates.get(f2)
        
        days_since_f1 = None
        days_since_f2 = None
        
        if current_date is not None:
            if last_date_f1 is not None:
                days_since_f1 = (current_date - last_date_f1).days
            if last_date_f2 is not None:
                days_since_f2 = (current_date - last_date_f2).days
        
        # Apply multiphase decay if enabled (Feature 1)
        if use_multiphase_decay and current_date is not None:
            r1_adjusted = apply_multiphase_decay(
                r1_adjusted, days_since_f1,
                multiphase_decay_params["quick_succession_days"],
                multiphase_decay_params["quick_succession_bump"],
                multiphase_decay_params["decay_days"],
                multiphase_decay_params["multiphase_decay_rate"]
            )
            r2_adjusted = apply_multiphase_decay(
                r2_adjusted, days_since_f2,
                multiphase_decay_params["quick_succession_days"],
                multiphase_decay_params["quick_succession_bump"],
                multiphase_decay_params["decay_days"],
                multiphase_decay_params["multiphase_decay_rate"]
            )
        # Apply standard decay if enabled (legacy mode)
        elif decay_mode != "none" and current_date is not None and decay_rate > 0:
            r1_adjusted = apply_decay(r1, days_since_f1, decay_rate, min_days, decay_mode)
            r2_adjusted = apply_decay(r2, days_since_f2, decay_rate, min_days, decay_mode)
        
        # Apply weight adjustment if enabled (Feature 2)
        # Compute weight change data once and reuse for both precomp and postcomp
        f1_moved_up = False
        f2_moved_up = False
        if use_weight_adjust:
            # Get current weight for both fighters
            w1 = row.get("weight_stat")
            if pd.isna(w1):
                w1 = row.get("weight_of_fight")
            w2 = row.get("opp_weight_stat")
            if pd.isna(w2):
                w2 = row.get("weight_of_fight")
            
            # Detect weight changes
            f1_moved_up, _, _ = detect_weight_change(f1, current_date, w1, weight_history)
            f2_moved_up, _, _ = detect_weight_change(f2, current_date, w2, weight_history)
            
            # Apply precomp penalty for moving up
            if f1_moved_up:
                r1_adjusted *= weight_adjust_params["weight_up_precomp_penalty"]
            if f2_moved_up:
                r2_adjusted *= weight_adjust_params["weight_up_precomp_penalty"]

        # Base expected scores (use adjusted ratings for prediction)
        e1 = calculate_expected_value(r1_adjusted, r2_adjusted, denominator)
        e2 = calculate_expected_value(r2_adjusted, r1_adjusted, denominator)

        # Method of victory factor
        if mov_params is not None:
            factor = mov_factor(row, mov_params)
        else:
            factor = 1.0

        k_eff = k * factor
        
        # Reduce K-factor for draws (draws are less decisive)
        if is_draw:
            k_eff = k_eff * draw_k_factor

        # Update ratings using the original (non-decayed) ratings
        # Decay only affects predictions, not the stored postcomp_elo
        r1_new = r1 + k_eff * (res - e1)
        r2_new = r2 + k_eff * ((1 - res) - e2)
        
        # Apply weight adjustment postcomp bonus (Feature 2) if fighter won
        # Uses the f1_moved_up/f2_moved_up values computed earlier
        if use_weight_adjust:
            # Apply postcomp bonus for winning when moved up
            if f1_moved_up and res == 1:  # Fighter 1 won
                r1_new *= weight_adjust_params["weight_up_postcomp_bonus"]
            if f2_moved_up and res == 0:  # Fighter 2 won
                r2_new *= weight_adjust_params["weight_up_postcomp_bonus"]

        ratings[f1], ratings[f2] = r1_new, r2_new
        
        # Update last fight dates for both fighters
        if current_date is not None:
            last_fight_dates[f1] = current_date
            last_fight_dates[f2] = current_date
        
        # Store original precomp_elo values (unchanged for reference)
        pre.append(r1)
        post.append(r1_new)
        opp_pre.append(r2)
        opp_post.append(r2_new)
        
        # Store adjusted precomp_elo values (for analysis)
        adjusted_pre.append(r1_adjusted)
        adjusted_opp_pre.append(r2_adjusted)

    df["precomp_elo"] = pre
    df["postcomp_elo"] = post
    df["opp_precomp_elo"] = opp_pre
    df["opp_postcomp_elo"] = opp_post
    
    # Add adjusted precomp_elo columns if any adjustment was applied
    if decay_mode != "none" or use_multiphase_decay or use_weight_adjust:
        df["adjusted_precomp_elo"] = adjusted_pre
        df["adjusted_opp_precomp_elo"] = adjusted_opp_pre
    
    return df


# =========================
# Accuracy helpers
# =========================



def compute_fight_predictions(df):
    """
    Build predictions from pre-fight Elo.
    Only counts fights where:
      - result is 0 or 1
      - precomp_elo != opp_precomp_elo
      - both fighters have at least one prior fight
    """
    hist = build_fighter_history(df)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    out = []

    for _, r in df.iterrows():
        d = r["DATE"]
        if pd.isna(d) or r["result"] not in (0, 1):
            continue
        if r["precomp_elo"] == r["opp_precomp_elo"]:
            continue
        if not has_prior_history(first_dates, r["FIGHTER"], d):
            continue
        if not has_prior_history(first_dates, r["opp_FIGHTER"], d):
            continue

        # Use precomp_elo for predictions (rating before the fight)
        pred = int(r["precomp_elo"] > r["opp_precomp_elo"])
        out.append(
            {
                "date": d,
                "prediction": pred,
                "result": int(r["result"]),
                "correct": int(pred == r["result"]),
            }
        )

    return pd.DataFrame(out)


def elo_accuracy(df, cutoff_date=None):
    preds = compute_fight_predictions(df)
    if preds.empty:
        return None, 0.0, 0

    acc_all = preds["correct"].mean()

    if cutoff_date is None:
        return acc_all, None, 0

    future = preds[preds["date"] > cutoff_date]
    if future.empty:
        return acc_all, None, 0

    acc_future = future["correct"].mean()
    return acc_all, acc_future, len(future)


# =========================
# OOS helpers
# =========================

def latest_ratings_from_trained_df(df, base_elo=1500, as_of_date=None):
    """
    Build {fighter_name: latest_post_fight_elo} from df with Elo columns.
    Uses both FIGHTER and opp_FIGHTER sides.
    
    If as_of_date is provided, only uses fights before that date to get ratings.
    Returns the postcomp_elo from each fighter's LAST fight before as_of_date.
    """
    if as_of_date is not None:
        df = df[df["DATE"] < as_of_date].copy()
    
    # Sort by date to ensure we get the latest rating for each fighter
    if "DATE" in df.columns:
        df = df.sort_values("DATE").reset_index(drop=True)
    
    ratings = {}
    for _, r in df.iterrows():
        # Use postcomp_elo (rating AFTER the fight)
        # Later fights will overwrite earlier ones, giving us the latest rating
        ratings[r["FIGHTER"]] = r["postcomp_elo"]
        ratings[r["opp_FIGHTER"]] = r["opp_postcomp_elo"]
    return ratings


def build_training_bout_counts(df):
    """
    Count how many bouts each fighter has in the training data.
    This is used to enforce the "must have at least N prior fights" rule
    on the OOS evaluation.
    """
    counts = {}
    for _, r in df.iterrows():
        f1 = r["FIGHTER"]
        f2 = r["opp_FIGHTER"]
        counts[f1] = counts.get(f1, 0) + 1
        counts[f2] = counts.get(f2, 0) + 1
    return counts


def test_out_of_sample_accuracy(
    df_trained_with_elo,
    test_df,
    base_elo=1500,
    verbose=False,
    gap_threshold=None,
    min_train_bouts=1,
):
    """
    Evaluate OOS accuracy using frozen ratings from df_trained_with_elo.
    Does not update ratings during OOS.

    Only counts OOS fights where BOTH fighters have at least min_train_bouts
    fights in the training data.
    """
    bout_counts = build_training_bout_counts(df_trained_with_elo)
    
    # Build set of all fighter names in training data for name matching
    all_training_fighters = set()
    for _, r in df_trained_with_elo.iterrows():
        all_training_fighters.add(r["FIGHTER"])
        all_training_fighters.add(r["opp_FIGHTER"])

    tdf = test_df.copy()
    tdf["result"] = pd.to_numeric(tdf["result"], errors="coerce")
    tdf["DATE"] = pd.to_datetime(tdf["date"])

    hits_all, total_all = 0, 0
    hits_gap, total_gap = 0, 0

    raw_count = 0
    used_count = 0
    skipped_boutcount = 0

    for _, row in tdf.iterrows():
        f1_test = row["fighter"]
        f2_test = row["opp_fighter"]
        res = row["result"]
        test_date = row["DATE"]

        if res not in (0, 1):
            continue

        raw_count += 1

        # Try to match fighter names (handles name variations)
        f1 = find_fighter_match(f1_test, all_training_fighters) or f1_test
        f2 = find_fighter_match(f2_test, all_training_fighters) or f2_test

        # Get total bout counts for both fighters from training data (for filtering)
        f1_bouts = bout_counts.get(f1, 0)
        f2_bouts = bout_counts.get(f2, 0)
        
        # Calculate precomp_boutcount for this specific test fight date
        # (how many fights each fighter had BEFORE this test date)
        f1_precomp = len(df_trained_with_elo[
            (df_trained_with_elo["DATE"] < test_date) & 
            ((df_trained_with_elo["FIGHTER"] == f1) | (df_trained_with_elo["opp_FIGHTER"] == f1))
        ])
        f2_precomp = len(df_trained_with_elo[
            (df_trained_with_elo["DATE"] < test_date) & 
            ((df_trained_with_elo["FIGHTER"] == f2) | (df_trained_with_elo["opp_FIGHTER"] == f2))
        ])

        # Bout count rule: both fighters must have at least min_train_bouts fights in training
        if f1_bouts < min_train_bouts or f2_bouts < min_train_bouts:
            skipped_boutcount += 1
            if verbose:
                name_note = f" (matched: {f1}/{f2})" if (f1 != f1_test or f2 != f2_test) else ""
                print(
                    f"{row['DATE'].date()} | {f1_test} vs {f2_test}{name_note} | "
                    f"precomp_boutcount: {f1_precomp}/{f2_precomp} | "
                    f"train_bouts: {f1_bouts}/{f2_bouts} -> SKIPPED (train_bouts < {min_train_bouts})"
                )
            continue

        used_count += 1

        # Get postcomp_elo ratings as of this test date (ratings AFTER their last fight before test date)
        rating_lookup = latest_ratings_from_trained_df(df_trained_with_elo, base_elo=base_elo, as_of_date=test_date)
        r1 = rating_lookup.get(f1, base_elo)
        r2 = rating_lookup.get(f2, base_elo)
        pred = int(r1 > r2)
        correct = int(pred == res)

        hits_all += correct
        total_all += 1

        if gap_threshold is not None and abs(r1 - r2) >= gap_threshold:
            hits_gap += correct
            total_gap += 1

        if verbose:
            name_note = f" (matched: {f1}/{f2})" if (f1 != f1_test or f2 != f2_test) else ""
            print(
                f"{row['DATE'].date()} | {f1_test} vs {f2_test}{name_note} | "
                f"precomp_boutcount: {f1_precomp}/{f2_precomp} | "
                f"train_bouts: {f1_bouts}/{f2_bouts} | "
                f"postcomp_elo: {r1:.1f}/{r2:.1f} -> pred={pred} res={int(res)} {'✓' if correct else '✗'}"
            )

    acc_all = hits_all / total_all if total_all else None
    acc_gap = hits_gap / total_gap if total_gap else None

    if verbose:
        print(f"\nOOS fights with labels: {raw_count}")
        print(f"OOS used after boutcount rule (min_train_bouts={min_train_bouts}): {used_count}")
        print(f"OOS skipped by boutcount rule: {skipped_boutcount}")

        if gap_threshold is not None:
            print(f"\nGAP FILTER ACTIVE: abs(Elo diff) >= {gap_threshold}")
            print(f"gap fights counted: {total_gap}")
            print(f"gap accuracy: {acc_gap}")

    return acc_all


# =========================
# Genetic algorithm over k and MOV weights
# =========================

def _calculate_oos_accuracy_worker(args):
    """Module-level worker function for parallel OOS calculation (must be at module level for multiprocessing)"""
    params_dict, train_data, test_data = args
    params = params_dict["params"]
    mov_params = {
        "w_ko": params["w_ko"],
        "w_sub": params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    
    # Train on all data before test dates
    df_trained = run_basic_elo(train_data.copy(), k=params["k"], mov_params=mov_params)
    
    # Calculate OOS accuracy
    oos_acc = test_out_of_sample_accuracy(
        df_trained,
        test_data,
        base_elo=1500,
        verbose=False,
        min_train_bouts=1,
    )
    return params_dict["index"], oos_acc

# Base parameter bounds (always included)
BASE_PARAM_BOUNDS = {
    "k":          (10.0, 500.0),
    "w_ko":       (1.0, 2.0),
    "w_sub":      (1.0, 2.0),
    "w_udec":     (0.8, 1.2),
    "w_sdec":     (0.5, 1.1),
    "w_mdec":     (0.7, 1.2),
    "decay_rate": (0.0, 0.01),    # Decay rate per day
    "min_days":   (0, 365),       # Minimum days before decay starts
}

# Multiphase decay parameters (Feature 1)
# Note: quick_succession_days max (89) < decay_days min (91) to avoid boundary overlap
MULTIPHASE_DECAY_BOUNDS = {
    "quick_succession_days": (7.0, 89.0),    # Days threshold for quick succession bump
    "quick_succession_bump": (1.00, 1.15),   # Multiplier for recent fighters (> 1.0 = boost)
    "decay_days":            (91.0, 365.0),  # Days threshold for decay to start
    "multiphase_decay_rate": (0.0005, 0.015), # Exponential decay rate
}

# Weight change parameters (Feature 2)
WEIGHT_ADJUST_BOUNDS = {
    "weight_up_precomp_penalty":  (0.80, 0.98),  # Penalty for moving up in weight (< 1.0)
    "weight_up_postcomp_bonus":   (1.05, 1.25),  # Bonus for winning in higher weight class
}

# Elo denomination parameter (Feature 3)
ELO_DENOM_BOUNDS = {
    "elo_denom": (200.0, 800.0),  # The denominator in Elo expected value calculation
}

# Default PARAM_BOUNDS for backward compatibility
PARAM_BOUNDS = BASE_PARAM_BOUNDS.copy()


def get_param_bounds(multiphase_decay=False, weight_adjust=False, optimize_elo_denom=False):
    """
    Get parameter bounds based on active feature flags.
    
    Args:
        multiphase_decay: If True, includes multiphase decay parameters
        weight_adjust: If True, includes weight adjustment parameters
        optimize_elo_denom: If True, includes elo_denom parameter
    
    Returns:
        dict: Parameter bounds for GA optimization
    """
    bounds = BASE_PARAM_BOUNDS.copy()
    
    if multiphase_decay:
        bounds.update(MULTIPHASE_DECAY_BOUNDS)
    
    if weight_adjust:
        bounds.update(WEIGHT_ADJUST_BOUNDS)
    
    if optimize_elo_denom:
        bounds.update(ELO_DENOM_BOUNDS)
    
    return bounds


def random_param_value(key, param_bounds=None):
    if param_bounds is None:
        param_bounds = PARAM_BOUNDS
    lo, hi = param_bounds[key]
    return random.uniform(lo, hi)


def random_params(include_decay=False, multiphase_decay=False, weight_adjust=False, 
                  optimize_elo_denom=False, param_bounds=None):
    """
    Generate random parameter values for GA optimization.
    
    Args:
        include_decay: If True, includes decay_rate and min_days parameters
        multiphase_decay: If True, includes multiphase decay parameters
        weight_adjust: If True, includes weight adjustment parameters
        optimize_elo_denom: If True, includes elo_denom parameter
        param_bounds: Optional custom parameter bounds dict
    
    Returns:
        dict: Random parameter values
    """
    if param_bounds is None:
        param_bounds = get_param_bounds(multiphase_decay, weight_adjust, optimize_elo_denom)
    
    params = {
        "k":      random_param_value("k", param_bounds),
        "w_ko":   random_param_value("w_ko", param_bounds),
        "w_sub":  random_param_value("w_sub", param_bounds),
        "w_udec": random_param_value("w_udec", param_bounds),
        "w_sdec": random_param_value("w_sdec", param_bounds),
        "w_mdec": random_param_value("w_mdec", param_bounds),
    }
    
    if include_decay:
        params["decay_rate"] = random_param_value("decay_rate", param_bounds)
        params["min_days"] = int(random_param_value("min_days", param_bounds))
    
    if multiphase_decay:
        params["quick_succession_days"] = int(random_param_value("quick_succession_days", param_bounds))
        params["quick_succession_bump"] = random_param_value("quick_succession_bump", param_bounds)
        params["decay_days"] = int(random_param_value("decay_days", param_bounds))
        params["multiphase_decay_rate"] = random_param_value("multiphase_decay_rate", param_bounds)
    
    if weight_adjust:
        params["weight_up_precomp_penalty"] = random_param_value("weight_up_precomp_penalty", param_bounds)
        params["weight_up_postcomp_bonus"] = random_param_value("weight_up_postcomp_bonus", param_bounds)
    
    if optimize_elo_denom:
        params["elo_denom"] = random_param_value("elo_denom", param_bounds)
    
    return params


def clip_param(key, value, param_bounds=None):
    if param_bounds is None:
        param_bounds = PARAM_BOUNDS
    if key not in param_bounds:
        return value
    lo, hi = param_bounds[key]
    return max(lo, min(hi, value))


def evaluate_params(df, cutoff_date, params, decay_mode="none"):
    """
    Run Elo with given params and return future accuracy.
    Training accuracy already uses boutcount rule through compute_fight_predictions.
    
    Args:
        df: Training data
        cutoff_date: Date to split training/validation
        params: Dict with k, w_ko, w_sub, w_udec, w_sdec, w_mdec, and optionally decay_rate, min_days
        decay_mode: "linear", "exponential", or "none" (default "none")
    """
    mov_params = {
        "w_ko":   params["w_ko"],
        "w_sub":  params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    k = params["k"]
    
    # Get decay parameters if present
    decay_rate = params.get("decay_rate", 0.0)
    min_days = params.get("min_days", 180)

    trial = run_basic_elo(df, k=k, mov_params=mov_params, 
                          decay_mode=decay_mode, decay_rate=decay_rate, min_days=min_days)
    _, acc_future, _ = elo_accuracy(trial, cutoff_date)

    if acc_future is None:
        return 0.0
    return acc_future


# =========================
# ROI-based fitness helpers
# =========================

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


def build_bidirectional_odds_lookup(odds_df):
    """
    Build a bidirectional odds lookup dictionary from odds data.
    
    This creates lookup entries for both (fighter, opponent, date) and
    (opponent, fighter, date) so that odds can be found regardless of
    which fighter has the higher Elo rating.
    
    IMPORTANT: The odds_df should contain rows for BOTH fighters in each fight,
    with each fighter's correct odds in their respective row. For example:
    - Row 1: FIGHTER='Fighter A', opp_FIGHTER='Fighter B', avg_odds=-150 (A is favorite)
    - Row 2: FIGHTER='Fighter B', opp_FIGHTER='Fighter A', avg_odds=+120 (B is underdog)
    
    If odds_df only has one row per fight, the reverse key will fall back to
    using the same odds, which may not be accurate for ROI calculations.
    
    Args:
        odds_df: DataFrame with DATE, FIGHTER, opp_FIGHTER, and avg_odds columns
    
    Returns:
        dict: Mapping (fighter, opponent, date_str) -> avg_odds value for that fighter
    """
    odds_lookup = {}
    
    for _, row in odds_df.iterrows():
        if pd.notna(row.get('avg_odds')):
            date_str = str(pd.to_datetime(row['DATE']).date())
            key = (row['FIGHTER'], row['opp_FIGHTER'], date_str)
            odds_lookup[key] = row['avg_odds']
    
    return odds_lookup


def compute_prediction_metrics(df_with_elo, odds_df, lookback_days=0):
    """
    Compute accuracy, log loss, and Brier score for Elo predictions.
    
    Args:
        df_with_elo: DataFrame with Elo ratings already calculated
        odds_df: Odds data for ROI calculations
        lookback_days: Number of days to look back (0 or None for all data, default 0)
    
    Returns:
        dict: Contains 'accuracy', 'log_loss', 'brier_score', 'total_predictions'
    """
    # Filter to lookback period if requested
    if lookback_days and lookback_days > 0:
        max_date = df_with_elo["DATE"].max()
        cutoff_date = max_date - pd.Timedelta(days=lookback_days)
        test_df = df_with_elo[df_with_elo["DATE"] > cutoff_date].copy()
    else:
        test_df = df_with_elo.copy()
    
    # Build fighter history for prior fight check
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    predictions = []
    actuals = []
    processed_fights = set()
    
    for _, row in test_df.iterrows():
        result = row["result"]
        if result not in (0, 1):
            continue
        if pd.isna(row["DATE"]):
            continue
        
        fighter = row["FIGHTER"]
        opponent = row["opp_FIGHTER"]
        date_str = str(row["DATE"].date())
        
        # Create unique fight key to avoid double counting
        fight_key = tuple(sorted([fighter, opponent])) + (date_str,)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)
        
        # Skip if equal Elo ratings
        if row["precomp_elo"] == row["opp_precomp_elo"]:
            continue
        
        # Skip if either fighter has no prior history
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        # Calculate predicted probability using Elo formula
        elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
        
        # Actual outcome (1 if fighter won, 0 if opponent won)
        actual = int(result)
        
        predictions.append(pred_prob)
        actuals.append(actual)
    
    if len(predictions) == 0:
        return {
            'accuracy': None,
            'log_loss': None,
            'brier_score': None,
            'total_predictions': 0
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Accuracy: % of correct predictions (predict winner based on higher prob)
    pred_winners = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_winners == actuals)
    
    # Log loss: -mean(y*log(p) + (1-y)*log(1-p))
    # Clip predictions to avoid log(0)
    eps = 1e-15
    predictions_clipped = np.clip(predictions, eps, 1 - eps)
    log_loss = -np.mean(
        actuals * np.log(predictions_clipped) + 
        (1 - actuals) * np.log(1 - predictions_clipped)
    )
    
    # Brier score: mean((p - y)^2)
    brier_score = np.mean((predictions - actuals) ** 2)
    
    return {
        'accuracy': accuracy,
        'log_loss': log_loss,
        'brier_score': brier_score,
        'total_predictions': len(predictions)
    }


def compute_extended_roi_metrics(bet_records):
    """
    Compute extended ROI metrics from a list of bet records.
    
    Args:
        bet_records: List of dicts with keys: date, profit, bet_amount, bet_won
    
    Returns:
        dict: Contains 'trend', 'sharpe_ratio', 'min_roi', 'max_roi', 'win_rate', 
              'num_bets', 'total_wagered', 'total_profit'
    """
    if not bet_records:
        return {
            'trend': None,
            'sharpe_ratio': None,
            'min_roi': None,
            'max_roi': None,
            'win_rate': None,
            'num_bets': 0,
            'total_wagered': 0.0,
            'total_profit': 0.0
        }
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(bet_records)
    
    num_bets = len(df)
    total_wagered = df['bet_amount'].sum()
    total_profit = df['profit'].sum()
    
    # Win rate
    wins = df['bet_won'].sum()
    win_rate = wins / num_bets if num_bets > 0 else 0.0
    
    # Group by date to calculate per-period metrics
    daily_df = df.groupby('date').agg({
        'profit': 'sum',
        'bet_amount': 'sum'
    }).reset_index()
    daily_df['roi'] = (daily_df['profit'] / daily_df['bet_amount']) * 100
    
    # Calculate cumulative ROI
    daily_df['cumulative_profit'] = daily_df['profit'].cumsum()
    daily_df['cumulative_wagered'] = daily_df['bet_amount'].cumsum()
    daily_df['cumulative_roi'] = (daily_df['cumulative_profit'] / daily_df['cumulative_wagered']) * 100
    
    # Trend: linear regression slope of cumulative ROI over time
    trend = None
    if len(daily_df) >= 2:
        x = np.arange(len(daily_df))
        y = daily_df['cumulative_roi'].values
        # Simple linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator > 0:
            trend = numerator / denominator  # %/day (or %/period)
    
    # Sharpe Ratio: (mean return / std return) * sqrt(252) for annualized
    # Using daily ROI as returns
    sharpe_ratio = None
    if len(daily_df) >= 2:
        returns = daily_df['roi'].values
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return > 0:
            # Annualized Sharpe (assuming ~52 betting periods/year for weekly events)
            sharpe_ratio = (mean_return / std_return) * math.sqrt(52)
    
    # Min and Max ROI per period
    min_roi = daily_df['roi'].min() if len(daily_df) > 0 else None
    max_roi = daily_df['roi'].max() if len(daily_df) > 0 else None
    
    return {
        'trend': trend,
        'sharpe_ratio': sharpe_ratio,
        'min_roi': min_roi,
        'max_roi': max_roi,
        'win_rate': win_rate,
        'num_bets': num_bets,
        'total_wagered': total_wagered,
        'total_profit': total_profit
    }


def evaluate_params_roi(df, odds_df, params, lookback_days=0, return_extended=False, decay_mode="none",
                        multiphase_decay=False, weight_adjust=False, optimize_elo_denom=False,
                        weight_history=None):
    """
    Run Elo with given params and return ROI% for fights with available odds.
    
    This function:
    1. Trains Elo model on ALL historical data using the candidate parameters
    2. Calculates ROI by simulating $1 bets on the higher-rated fighter for fights with odds
    3. Returns ROI% as the fitness metric
    
    Args:
        df: Training data (all historical fight data with DATE, FIGHTER, opp_FIGHTER, result)
        odds_df: Odds data with DATE, FIGHTER, opp_FIGHTER, avg_odds columns
        params: Dict with k, w_ko, w_sub, w_udec, w_sdec, w_mdec, and optionally decay_rate, min_days
        lookback_days: Number of days to look back for ROI calculation (default 0 = all data)
                       Set to 0 or None to use all fights with available odds
        return_extended: If True, return dict with extended metrics instead of just ROI
        decay_mode: "linear", "exponential", or "none" (default "none")
        multiphase_decay: If True, use multiphase decay parameters from params
        weight_adjust: If True, use weight adjustment parameters from params
        optimize_elo_denom: If True, use elo_denom parameter from params
        weight_history: Pre-built fighter weight history (from build_fighter_weight_history)
    
    Returns:
        If return_extended=False: float: ROI percentage (0 if no valid bets)
        If return_extended=True: dict with roi_percent, trend, sharpe_ratio, min_roi, max_roi,
                                 win_rate, num_bets, accuracy, log_loss, brier_score, df_with_elo
    """
    mov_params = {
        "w_ko":   params["w_ko"],
        "w_sub":  params["w_sub"],
        "w_udec": params["w_udec"],
        "w_sdec": params["w_sdec"],
        "w_mdec": params["w_mdec"],
    }
    k = params["k"]
    
    # Get decay parameters if present
    decay_rate = params.get("decay_rate", 0.0)
    min_days = params.get("min_days", 180)
    
    # Get elo_denom if optimize_elo_denom is enabled (Feature 3)
    elo_denom = params.get("elo_denom", 400) if optimize_elo_denom else 400
    
    # Prepare multiphase decay params (Feature 1)
    multiphase_decay_params = None
    if multiphase_decay:
        multiphase_decay_params = {
            "quick_succession_days": params.get("quick_succession_days", 30),
            "quick_succession_bump": params.get("quick_succession_bump", 1.05),
            "decay_days": params.get("decay_days", 180),
            "multiphase_decay_rate": params.get("multiphase_decay_rate", 0.002),
        }
    
    # Prepare weight adjust params (Feature 2)
    weight_adjust_params = None
    if weight_adjust:
        weight_adjust_params = {
            "weight_up_precomp_penalty": params.get("weight_up_precomp_penalty", 0.95),
            "weight_up_postcomp_bonus": params.get("weight_up_postcomp_bonus", 1.10),
        }
    
    # Train Elo on ALL historical data
    df_with_elo = run_basic_elo(
        df.copy(), k=k, mov_params=mov_params, denominator=elo_denom,
        decay_mode=decay_mode, decay_rate=decay_rate, min_days=min_days,
        multiphase_decay_params=multiphase_decay_params,
        weight_adjust_params=weight_adjust_params,
        weight_history=weight_history
    )
    
    # Filter to recent fights if lookback_days is specified (>0)
    if lookback_days and lookback_days > 0:
        max_date = df_with_elo["DATE"].max()
        cutoff_date = max_date - pd.Timedelta(days=lookback_days)
        test_df = df_with_elo[df_with_elo["DATE"] > cutoff_date].copy()
    else:
        # Use all fights (will filter to those with odds available during ROI calculation)
        test_df = df_with_elo.copy()
    
    # Build bidirectional odds lookup
    odds_lookup = build_bidirectional_odds_lookup(odds_df)
    
    # Build fighter history for prior fight check
    hist = build_fighter_history(df_with_elo)
    first_dates = hist.groupby("fighter")["date"].min().to_dict()
    
    # Calculate ROI - track bet records for extended metrics
    total_wagered = 0.0
    total_profit = 0.0
    processed_fights = set()  # Track unique fights to avoid double counting
    bet_records = []  # Track individual bets for extended metrics
    
    for _, row in test_df.iterrows():
        # Skip invalid results
        result = row["result"]
        if result not in (0, 1):
            continue
        if pd.isna(row["DATE"]):
            continue
        
        fighter = row["FIGHTER"]
        opponent = row["opp_FIGHTER"]
        date_str = str(row["DATE"].date())
        
        # Create a unique key for each fight (sorted fighter names + date)
        fight_key = tuple(sorted([fighter, opponent])) + (date_str,)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)
        
        # Skip if equal Elo ratings
        if row["precomp_elo"] == row["opp_precomp_elo"]:
            continue
        
        # Skip if either fighter has no prior history
        if not has_prior_history(first_dates, fighter, row["DATE"]):
            continue
        if not has_prior_history(first_dates, opponent, row["DATE"]):
            continue
        
        # Determine the higher Elo fighter
        fighter_has_higher_elo = row["precomp_elo"] > row["opp_precomp_elo"]
        
        # We always bet on the higher Elo fighter
        if fighter_has_higher_elo:
            # FIGHTER has higher Elo - look up their odds
            bet_on = fighter
            bet_against = opponent
            odds_key = (fighter, opponent, date_str)
            # result=1 means FIGHTER won (our bet won)
            bet_won = (result == 1)
        else:
            # Opponent has higher Elo - look up opponent's odds
            bet_on = opponent
            bet_against = fighter
            odds_key = (opponent, fighter, date_str)
            # result=0 means FIGHTER lost, so opponent (our bet) won
            bet_won = (result == 0)
        
        # Look up odds for the fighter we're betting on
        if odds_key not in odds_lookup:
            continue
        
        bet_odds = odds_lookup[odds_key]
        decimal_odds = american_odds_to_decimal(bet_odds)
        if decimal_odds is None:
            continue
        
        # Simulate $1 bet
        bet_amount = 1.0
        total_wagered += bet_amount
        
        if bet_won:
            payout = bet_amount * decimal_odds
            profit = payout - bet_amount
        else:
            profit = -bet_amount
        
        total_profit += profit
        
        # Track bet record for extended metrics
        bet_records.append({
            'date': row["DATE"],
            'profit': profit,
            'bet_amount': bet_amount,
            'bet_won': int(bet_won)
        })
    
    # Calculate ROI
    if total_wagered == 0:
        if return_extended:
            return {
                'roi_percent': 0.0,
                'trend': None,
                'sharpe_ratio': None,
                'min_roi': None,
                'max_roi': None,
                'win_rate': None,
                'num_bets': 0,
                'accuracy': None,
                'log_loss': None,
                'brier_score': None,
                'df_with_elo': df_with_elo,
                # Comprehensive metrics (new)
                'ece': None,
                'calibration_slope': None,
                'auc_roc': None,
                'consistency_variance': None,
                'max_drawdown': None,
                'comprehensive_metrics': None
            }
        return 0.0
    
    roi_percent = (total_profit / total_wagered) * 100
    
    if not return_extended:
        return roi_percent
    
    # Compute extended metrics
    extended = compute_extended_roi_metrics(bet_records)
    prediction_metrics = compute_prediction_metrics(df_with_elo, odds_df, lookback_days)
    
    # Compute comprehensive metrics (calibration, consistency, performance, betting)
    comprehensive = compute_comprehensive_metrics(df_with_elo, bet_records, odds_df)
    
    return {
        'roi_percent': roi_percent,
        'trend': extended['trend'],
        'sharpe_ratio': extended['sharpe_ratio'],
        'min_roi': extended['min_roi'],
        'max_roi': extended['max_roi'],
        'win_rate': extended['win_rate'],
        'num_bets': extended['num_bets'],
        'accuracy': prediction_metrics['accuracy'],
        'log_loss': prediction_metrics['log_loss'],
        'brier_score': prediction_metrics['brier_score'],
        'df_with_elo': df_with_elo,
        # Comprehensive metrics (new)
        'ece': comprehensive['calibration']['ece'],
        'calibration_slope': comprehensive['calibration']['calibration_slope'],
        'auc_roc': comprehensive['performance']['auc_roc'],
        'consistency_variance': comprehensive['consistency']['overall_variance'],
        'max_drawdown': comprehensive['betting']['drawdown']['max_drawdown'],
        'comprehensive_metrics': comprehensive
    }


def _calculate_roi_worker(args):
    """Module-level worker function for parallel ROI calculation (must be at module level for multiprocessing)"""
    params_dict, train_data, odds_data, lookback_days = args
    params = params_dict["params"]
    
    roi = evaluate_params_roi(train_data, odds_data, params, lookback_days=lookback_days)
    return params_dict["index"], roi


def ga_search_params_roi(
    df,
    odds_df,
    test_df=None,
    population_size=30,
    generations=30,
    lookback_days=0,
    seed=42,
    return_all_results=False,
    verbose=True,
    fitness_weights=None,
    decay_mode="none",
    multiphase_decay=False,
    weight_adjust=False,
    optimize_elo_denom=False,
):
    """
    Full GA search over k and method of victory weights using ROI as fitness.
    
    This function optimizes parameters to maximize ROI on fights with available odds:
    1. Load all historical fight data
    2. For each candidate parameter set:
       - Train Elo model on ALL historical data using the candidate parameters
       - Calculate ROI by simulating $1 bets on the higher-rated fighter for fights with odds
       - Use ROI% as the fitness metric (or weighted combination with trend/sharpe)
    3. Evolve parameters to maximize fitness
    
    Args:
        df: All historical fight data (will be used for training Elo)
        odds_df: Odds data with avg_odds column (ROI calculated on all fights with odds)
        test_df: Optional test data for OOS evaluation tracking (does not affect fitness)
        population_size: Number of individuals in each generation
        generations: Number of generations to evolve
        lookback_days: Number of days to look back for ROI calculation (default 0 = all data with odds)
        seed: Random seed for reproducibility (None for random)
        return_all_results: If True, returns list of all generation results
        verbose: If True, prints progress for each generation
        fitness_weights: Optional dict with weights for multi-objective fitness:
                        {'roi': 0.6, 'trend': 0.3, 'sharpe': 0.1}
                        If None, uses ROI only. Values are normalized internally.
        decay_mode: "linear", "exponential", or "none" (default "none")
                    If not "none", decay_rate and min_days will be included in optimization
        multiphase_decay: If True, use multiphase decay feature (--multiphase-decay on)
        weight_adjust: If True, use weight adjustment feature (--weight-adjust on)
        optimize_elo_denom: If True, optimize elo_denom parameter (--optimize-elo-denom on)
    
    Returns:
        If return_all_results=False: best_params, best_fitness
        If return_all_results=True: best_params, best_fitness, all_results
    """
    include_decay = decay_mode != "none"
    
    # Get parameter bounds based on active features
    param_bounds = get_param_bounds(multiphase_decay, weight_adjust, optimize_elo_denom)
    
    if seed is not None:
        random.seed(seed)
    
    # Build weight history if weight_adjust is enabled
    weight_history = None
    if weight_adjust:
        if verbose:
            print("Building fighter weight history for weight adjustment feature...")
        weight_history = build_fighter_weight_history(df)
        if verbose:
            print(f"  Built weight history for {len(weight_history)} fighters")
    
    # Prepare odds data
    odds_df = odds_df.copy()
    odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
    
    # Calculate date range for odds data
    odds_min_date = odds_df["DATE"].min()
    odds_max_date = odds_df["DATE"].max()
    
    # Count unique fights with odds
    odds_fights = odds_df[['DATE', 'FIGHTER', 'opp_FIGHTER']].drop_duplicates()
    num_odds_fights = len(odds_fights) // 2  # Each fight appears twice (once per fighter)
    
    if verbose:
        if lookback_days and lookback_days > 0:
            max_date = df["DATE"].max()
            lookback_cutoff = max_date - pd.Timedelta(days=lookback_days)
            print(f"Optimizing ROI for fights from {lookback_cutoff.date()} to {max_date.date()} (last {lookback_days} days)")
            recent_fights = df[df["DATE"] > lookback_cutoff]
            # Count fights with odds in lookback period
            odds_in_lookback = odds_df[odds_df["DATE"] > lookback_cutoff]
            odds_fights_in_lookback = len(odds_in_lookback[['DATE', 'FIGHTER', 'opp_FIGHTER']].drop_duplicates()) // 2
            print(f"Fights in lookback period: {len(recent_fights)}")
            print(f"Fights with odds in lookback period: ~{odds_fights_in_lookback}")
            print(f"Total fights with odds (all time): ~{num_odds_fights}")
        else:
            print(f"Optimizing ROI on ALL fights with available odds")
            print(f"Odds data date range: {odds_min_date.date()} to {odds_max_date.date()}")
            print(f"Approximate fights with odds: {num_odds_fights}")
        print(f"Total historical fights: {len(df)}")
        if fitness_weights:
            print(f"Multi-objective fitness weights: {fitness_weights}")
        if decay_mode != "none":
            print(f"Decay mode: {decay_mode} (optimizing decay_rate and min_days)")
        
        # Log active feature flags
        active_features = []
        if multiphase_decay:
            active_features.append("multiphase-decay")
        if weight_adjust:
            active_features.append("weight-adjust")
        if optimize_elo_denom:
            active_features.append("optimize-elo-denom")
        if active_features:
            print(f"Active features: {', '.join(active_features)}")
        else:
            print("No advanced features active (baseline mode)")
    
    all_results = []
    
    def compute_fitness(extended_metrics):
        """Compute fitness from extended metrics, optionally using multi-objective weights.
        
        Supports comprehensive metric weights including:
        - roi: Return on investment
        - trend: ROI trend over time
        - sharpe: Sharpe ratio
        - calibration: Expected Calibration Error (lower is better)
        - consistency: Prediction consistency across subsets (lower variance is better)
        - auc: AUC-ROC score
        """
        roi = extended_metrics['roi_percent']
        
        if not fitness_weights:
            return roi
        
        # Default fitness weights for multi-metric optimization
        # (imported constants define: DEFAULT_ECE_THRESHOLD=0.1, DEFAULT_VARIANCE_THRESHOLD=0.01)
        default_weights = {'roi': 0.4, 'trend': 0.1, 'sharpe': 0.1, 'calibration': 0.15, 'consistency': 0.15, 'auc': 0.1}
        
        # Normalize weights
        total_weight = sum(fitness_weights.values())
        w_roi = fitness_weights.get('roi', default_weights['roi']) / total_weight
        w_trend = fitness_weights.get('trend', default_weights['trend']) / total_weight
        w_sharpe = fitness_weights.get('sharpe', default_weights['sharpe']) / total_weight
        w_calibration = fitness_weights.get('calibration', default_weights['calibration']) / total_weight
        w_consistency = fitness_weights.get('consistency', default_weights['consistency']) / total_weight
        w_auc = fitness_weights.get('auc', default_weights['auc']) / total_weight
        
        # Get metrics (use defaults if None)
        # ECE threshold: 0.1 is poor calibration, 0.01 is excellent
        # Variance threshold: 0.01 is inconsistent, 0.001 is excellent
        ECE_THRESHOLD = 0.1
        VARIANCE_THRESHOLD = 0.01
        
        trend = extended_metrics.get('trend') or 0
        sharpe = extended_metrics.get('sharpe_ratio') or 0
        ece = extended_metrics.get('ece') or ECE_THRESHOLD
        consistency_var = extended_metrics.get('consistency_variance') or VARIANCE_THRESHOLD
        auc = extended_metrics.get('auc_roc') or 0.5
        
        # Weighted combination (scale each to similar magnitude as ROI ~= -10 to +20)
        # ROI is already in percentage
        fitness = roi * w_roi
        
        # Trend: typically small (e.g., 0.1%/day), multiply by 10 to scale
        fitness += (trend * 10) * w_trend
        
        # Sharpe: typically 0-3, multiply by 5 to scale to ~0-15
        fitness += (sharpe * 5) * w_sharpe
        
        # Calibration: ECE 0.01 = excellent, 0.1 = poor
        # Invert and scale: lower ECE = higher fitness
        calibration_score = max(0, (ECE_THRESHOLD - ece) * 100)  # 0.01 ECE -> 9, 0.1 ECE -> 0
        fitness += calibration_score * w_calibration
        
        # Consistency: lower variance is better
        # variance 0.001 = excellent, 0.01 = poor
        consistency_score = max(0, (VARIANCE_THRESHOLD - consistency_var) * 1000)  # 0.001 -> 9, 0.01 -> 0
        fitness += consistency_score * w_consistency
        
        # AUC-ROC: 0.5 = random, 1.0 = perfect
        # Scale to ~0-20 range
        auc_score = (auc - 0.5) * 40  # 0.5 -> 0, 0.75 -> 10, 1.0 -> 20
        fitness += auc_score * w_auc
        
        return fitness
    
    def evaluate_individual(params):
        """Evaluate params and return fitness plus extended metrics."""
        extended = evaluate_params_roi(
            df, odds_df, params, lookback_days=lookback_days, 
            return_extended=True, decay_mode=decay_mode,
            multiphase_decay=multiphase_decay, weight_adjust=weight_adjust,
            optimize_elo_denom=optimize_elo_denom, weight_history=weight_history
        )
        fitness = compute_fitness(extended)
        return fitness, extended
    
    # Initialize population with diverse parameters
    population = []
    for i in range(population_size):
        p = random_params(
            include_decay=include_decay, multiphase_decay=multiphase_decay,
            weight_adjust=weight_adjust, optimize_elo_denom=optimize_elo_denom,
            param_bounds=param_bounds
        )
        fitness, extended = evaluate_individual(p)
        population.append({"params": p, "fitness": fitness, "extended": extended})
    
    best_ind = max(population, key=lambda ind: ind["fitness"])
    if verbose:
        fitnesses = [ind["fitness"] for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        ext = best_ind.get("extended", {})
        trend_str = f"{ext.get('trend', 0):.2f}%/day" if ext.get('trend') is not None else "N/A"
        sharpe_str = f"{ext.get('sharpe_ratio', 0):.2f}" if ext.get('sharpe_ratio') is not None else "N/A"
        win_rate_str = f"{ext.get('win_rate', 0)*100:.0f}%" if ext.get('win_rate') is not None else "N/A"
        num_bets = ext.get('num_bets', 0)
        print(
            f"Initial population: best ROI={best_ind['fitness']:.2f}%, "
            f"Trend={trend_str}, Sharpe={sharpe_str}, WinRate={win_rate_str}, Bets={num_bets}"
        )
    
    # GA loop
    for gen in range(generations):
        new_population = []
        
        # Elitism - keep top individual
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        elite = population[0].copy()
        new_population.append(elite)
        
        # Fill the rest
        while len(new_population) < population_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child_params = crossover(
                parent1, parent2, include_decay=include_decay,
                multiphase_decay=multiphase_decay, weight_adjust=weight_adjust,
                optimize_elo_denom=optimize_elo_denom
            )
            child_params = mutate(
                child_params, include_decay=include_decay,
                multiphase_decay=multiphase_decay, weight_adjust=weight_adjust,
                optimize_elo_denom=optimize_elo_denom, param_bounds=param_bounds
            )
            
            fitness, extended = evaluate_individual(child_params)
            new_population.append({"params": child_params, "fitness": fitness, "extended": extended})
        
        population = new_population
        gen_best = max(population, key=lambda ind: ind["fitness"])
        is_new_best = gen_best["fitness"] > best_ind["fitness"]
        if is_new_best:
            best_ind = gen_best.copy()
        
        # Calculate population statistics (needed for both verbose and return_all_results)
        fitnesses = [ind["fitness"] for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)
        max_fitness = max(fitnesses)
        
        if verbose:
            ext = gen_best.get("extended", {})
            roi = ext.get('roi_percent', gen_best['fitness'])
            trend_str = f"{ext.get('trend', 0):+.2f}%/day" if ext.get('trend') is not None else "N/A"
            sharpe_str = f"{ext.get('sharpe_ratio', 0):.2f}" if ext.get('sharpe_ratio') is not None else "N/A"
            win_rate_str = f"{ext.get('win_rate', 0)*100:.0f}%" if ext.get('win_rate') is not None else "N/A"
            num_bets = ext.get('num_bets', 0)
            
            # Include decay params in verbose output if enabled
            decay_str = ""
            if include_decay:
                decay_rate = gen_best['params'].get('decay_rate', 0)
                min_days_val = gen_best['params'].get('min_days', 180)
                decay_str = f", decay_rate={decay_rate:.5f}, min_days={min_days_val:.0f}"
            
            print(
                f"Gen {gen + 1:02d}, best ROI={roi:+.2f}%, Trend={trend_str}, "
                f"Sharpe={sharpe_str}, WinRate={win_rate_str}, Bets={num_bets}, "
                f"range=[{min_fitness:+.2f}%, {max_fitness:+.2f}%]"
            )
            
            # Print accuracy, log loss, brier score for best params
            accuracy = ext.get('accuracy')
            log_loss = ext.get('log_loss')
            brier_score = ext.get('brier_score')
            if accuracy is not None:
                print(
                    f"  Accuracy: {accuracy*100:.1f}%, LogLoss: {log_loss:.3f}, BrierScore: {brier_score:.3f}"
                )
            
            # Print comprehensive metrics if available
            ece = ext.get('ece')
            auc = ext.get('auc_roc')
            calib_slope = ext.get('calibration_slope')
            if ece is not None or auc is not None:
                ece_str = f"ECE={ece:.4f}" if ece is not None else "ECE=N/A"
                auc_str = f"AUC={auc:.3f}" if auc is not None else "AUC=N/A"
                slope_str = f"CalibSlope={calib_slope:.2f}" if calib_slope is not None else "CalibSlope=N/A"
                print(f"  {ece_str}, {auc_str}, {slope_str}")
            
            if is_new_best:
                print(f"  *** NEW BEST: k={gen_best['params']['k']:.1f}, ROI={roi:+.2f}%")
        
        if return_all_results:
            ext = gen_best.get("extended", {})
            gen_summary = {
                'generation': gen + 1,
                'best_fitness': gen_best['fitness'],
                'best_params': gen_best['params'].copy(),
                'population_avg_fitness': avg_fitness,
                'population_max_fitness': max_fitness,
                'population_min_fitness': min_fitness,
                # Extended metrics for best individual
                'best_roi_percent': ext.get('roi_percent'),
                'best_trend': ext.get('trend'),
                'best_sharpe_ratio': ext.get('sharpe_ratio'),
                'best_min_roi': ext.get('min_roi'),
                'best_max_roi': ext.get('max_roi'),
                'best_win_rate': ext.get('win_rate'),
                'best_num_bets': ext.get('num_bets'),
                'best_accuracy': ext.get('accuracy'),
                'best_log_loss': ext.get('log_loss'),
                'best_brier_score': ext.get('brier_score'),
                # Comprehensive metrics (new)
                'best_ece': ext.get('ece'),
                'best_calibration_slope': ext.get('calibration_slope'),
                'best_auc_roc': ext.get('auc_roc'),
                'best_consistency_variance': ext.get('consistency_variance'),
                'best_max_drawdown': ext.get('max_drawdown'),
            }
            all_results.append(gen_summary)
    
    if return_all_results:
        return best_ind["params"], best_ind["fitness"], all_results
    return best_ind["params"], best_ind["fitness"]


def calculate_oos_roi(df_trained, test_df, odds_df, verbose=False):
    """
    Calculate out-of-sample ROI on test data using pre-trained Elo ratings.
    
    Args:
        df_trained: DataFrame with Elo ratings already calculated
        test_df: Test data (e.g., past3_events.csv) with date, fighter, opp_fighter, result
        odds_df: Odds data with avg_odds column
        verbose: If True, prints details for each bet
    
    Returns:
        dict: ROI metrics including roi_percent, total_bets, total_profit, etc.
    """
    # Get all fighter names from training data for name matching
    all_training_fighters = set()
    for _, r in df_trained.iterrows():
        all_training_fighters.add(r["FIGHTER"])
        all_training_fighters.add(r["opp_FIGHTER"])
    
    # Get latest ratings from training data
    ratings = latest_ratings_from_trained_df(df_trained, base_elo=1500)
    
    # Build bidirectional odds lookup
    odds_lookup = build_bidirectional_odds_lookup(odds_df)
    
    # Process test data
    test_df_copy = test_df.copy()
    test_df_copy["DATE"] = pd.to_datetime(test_df_copy["date"])
    
    total_wagered = 0.0
    total_profit = 0.0
    total_bets = 0
    wins = 0
    processed_fights = set()
    
    for _, row in test_df_copy.iterrows():
        f1_test = row["fighter"]
        f2_test = row["opp_fighter"]
        res = row["result"]
        test_date = row["DATE"]
        
        if res not in (0, 1):
            continue
        
        # Create unique fight key
        fight_key = tuple(sorted([f1_test, f2_test])) + (str(test_date.date()),)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)
        
        # Try to match fighter names
        f1 = find_fighter_match(f1_test, all_training_fighters) or f1_test
        f2 = find_fighter_match(f2_test, all_training_fighters) or f2_test
        
        # Get ratings
        r1 = ratings.get(f1, 1500)
        r2 = ratings.get(f2, 1500)
        
        # Skip if equal ratings
        if r1 == r2:
            continue
        
        # Determine higher Elo fighter
        date_str = str(test_date.date())
        if r1 > r2:
            bet_on = f1
            bet_against = f2
            odds_key = (f1, f2, date_str)
            # Try reverse key too
            alt_odds_key = (f2, f1, date_str)
            bet_won = (res == 1)  # f1 won
        else:
            bet_on = f2
            bet_against = f1
            odds_key = (f2, f1, date_str)
            alt_odds_key = (f1, f2, date_str)
            bet_won = (res == 0)  # f1 lost, so f2 won
        
        # Look up odds
        if odds_key in odds_lookup:
            bet_odds = odds_lookup[odds_key]
        elif alt_odds_key in odds_lookup:
            bet_odds = odds_lookup[alt_odds_key]
        else:
            if verbose:
                print(f"Skipping {f1_test} vs {f2_test} on {date_str}: No odds found")
            continue
        
        decimal_odds = american_odds_to_decimal(bet_odds)
        if decimal_odds is None:
            continue
        
        # Simulate bet
        bet_amount = 1.0
        total_wagered += bet_amount
        total_bets += 1
        
        if bet_won:
            payout = bet_amount * decimal_odds
            profit = payout - bet_amount
            wins += 1
        else:
            profit = -bet_amount
        
        total_profit += profit
        
        if verbose:
            result_str = "WIN" if bet_won else "LOSS"
            print(f"{date_str}: {bet_on} ({r1 if bet_on == f1 else r2:.0f}) vs {bet_against} "
                  f"-> {result_str}, odds={bet_odds:+.0f}, profit=${profit:.2f}")
    
    if total_wagered == 0:
        return {
            'roi_percent': 0.0,
            'total_bets': 0,
            'total_wagered': 0.0,
            'total_profit': 0.0,
            'wins': 0,
            'accuracy': 0.0
        }
    
    roi_percent = (total_profit / total_wagered) * 100
    accuracy = wins / total_bets if total_bets > 0 else 0.0
    
    return {
        'roi_percent': roi_percent,
        'total_bets': total_bets,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'wins': wins,
        'accuracy': accuracy
    }


def tournament_select(population, k_tour=3):
    """
    Tournament selection: pick k individuals and return the best.
    population is a list of dicts with keys: params, fitness.
    
    If k_tour > len(population), uses len(population) instead.
    """
    k_tour = min(k_tour, len(population))
    contenders = random.sample(population, k_tour)
    contenders.sort(key=lambda ind: ind["fitness"], reverse=True)
    return contenders[0]


def crossover(parent1, parent2, crossover_rate=0.5, include_decay=False,
              multiphase_decay=False, weight_adjust=False, optimize_elo_denom=False):
    """
    Simple crossover.
    For each param, with probability crossover_rate take average, else take one parent.
    
    Args:
        parent1: First parent individual
        parent2: Second parent individual
        crossover_rate: Probability of averaging vs taking from one parent
        include_decay: If True, includes decay_rate and min_days in crossover
        multiphase_decay: If True, includes multiphase decay parameters
        weight_adjust: If True, includes weight adjustment parameters
        optimize_elo_denom: If True, includes elo_denom parameter
    """
    child_params = {}
    # Determine which keys to process (use all keys from parent1)
    keys_to_process = set(parent1["params"].keys())
    if not include_decay:
        keys_to_process -= {"decay_rate", "min_days"}
    
    for key in keys_to_process:
        v1 = parent1["params"].get(key)
        v2 = parent2["params"].get(key)
        
        # Skip if either parent doesn't have this key
        if v1 is None or v2 is None:
            continue
            
        if random.random() < crossover_rate:
            new_val = 0.5 * (v1 + v2)
            # Ensure day-count parameters are integers
            if key in ("min_days", "quick_succession_days", "decay_days"):
                new_val = int(new_val)
            child_params[key] = new_val
        else:
            val = random.choice([v1, v2])
            if key in ("min_days", "quick_succession_days", "decay_days"):
                val = int(val)
            child_params[key] = val
    return child_params


def mutate(params, mutation_rate=0.3, mutation_scale=0.1, include_decay=False,
           multiphase_decay=False, weight_adjust=False, optimize_elo_denom=False,
           param_bounds=None):
    """
    Gaussian mutation: for each param, with probability mutation_rate,
    add noise proportional to range * mutation_scale.
    
    Args:
        params: Dictionary of parameters
        mutation_rate: Probability of mutating each parameter
        mutation_scale: Scale factor for Gaussian noise
        include_decay: If True, includes decay_rate and min_days in mutation
        multiphase_decay: If True, includes multiphase decay parameters
        weight_adjust: If True, includes weight adjustment parameters
        optimize_elo_denom: If True, includes elo_denom parameter
        param_bounds: Custom parameter bounds dict (if None, uses get_param_bounds)
    """
    if param_bounds is None:
        param_bounds = get_param_bounds(multiphase_decay, weight_adjust, optimize_elo_denom)
    
    new_params = params.copy()
    mutated = False
    
    # Determine which keys to mutate (all keys present in params that have bounds)
    keys_to_mutate = set(new_params.keys())
    if not include_decay:
        keys_to_mutate -= {"decay_rate", "min_days"}
    
    # Day-count parameters that should be integers
    integer_params = {"min_days", "quick_succession_days", "decay_days"}
    
    for key in keys_to_mutate:
        if key not in param_bounds:
            continue
        if random.random() < mutation_rate:
            lo, hi = param_bounds[key]
            span = hi - lo
            noise = random.gauss(0.0, mutation_scale * span)
            new_val = new_params[key] + noise
            new_val = clip_param(key, new_val, param_bounds)
            # Ensure day-count parameters are integers
            if key in integer_params:
                new_val = int(new_val)
            new_params[key] = new_val
            mutated = True
    
    # Ensure at least one parameter is mutated to maintain diversity
    if not mutated and random.random() < 0.5:
        valid_keys = [k for k in keys_to_mutate if k in param_bounds]
        if valid_keys:
            key = random.choice(valid_keys)
            lo, hi = param_bounds[key]
            span = hi - lo
            noise = random.gauss(0.0, mutation_scale * span)
            new_val = new_params[key] + noise
            new_val = clip_param(key, new_val, param_bounds)
            if key in integer_params:
                new_val = int(new_val)
            new_params[key] = new_val
    return new_params


def ga_search_params(
    df,
    test_df=None,
    population_size=30,
    generations=30,
    cutoff_quantile=0.8,
    seed=42,
    return_all_results=False,
    verbose=True,
):
    """
    Full GA search over k and method of victory weights.
    
    Args:
        df: Training data
        test_df: Optional test data for OOS evaluation tracking (does not affect fitness)
        return_all_results: If True, returns a list of all generation results
        verbose: If True, prints progress for each generation
    
    Returns:
        If return_all_results=False: best_params, best_fitness, cutoff_date
        If return_all_results=True: best_params, best_fitness, cutoff_date, all_results
    """
    if seed is not None:
        random.seed(seed)

    cutoff_date = df["DATE"].quantile(cutoff_quantile)
    if verbose:
        print("Training end date:", cutoff_date)
    
    # Prepare test data for OOS evaluation if provided
    oos_test_start_date = None
    df_train_all_for_oos = None
    if test_df is not None:
        test_df_copy = test_df.copy()
        test_df_copy["DATE"] = pd.to_datetime(test_df_copy["date"])
        oos_test_start_date = test_df_copy["DATE"].min()
        df_train_all_for_oos = df[df["DATE"] < oos_test_start_date].copy()
        if verbose:
            print(f"OOS test start date: {oos_test_start_date}")

    all_results = []


    # Helper function to calculate OOS accuracy for a set of params (single-threaded fallback)
    def calculate_oos_accuracy(params):
        """Calculate OOS accuracy for given params (does not affect fitness)"""
        if test_df is None or df_train_all_for_oos is None:
            return None
        
        mov_params = {
            "w_ko": params["w_ko"],
            "w_sub": params["w_sub"],
            "w_udec": params["w_udec"],
            "w_sdec": params["w_sdec"],
            "w_mdec": params["w_mdec"],
        }
        
        # Train on all data before test dates
        df_trained = run_basic_elo(df_train_all_for_oos.copy(), k=params["k"], mov_params=mov_params)
        
        # Calculate OOS accuracy
        oos_acc = test_out_of_sample_accuracy(
            df_trained,
            test_df,
            base_elo=1500,
            verbose=False,
            min_train_bouts=1,
        )
        return oos_acc

    def calculate_oos_accuracy_batch(params_list, use_parallel=True):
        """Calculate OOS accuracy for a batch of params, optionally in parallel"""
        if test_df is None or df_train_all_for_oos is None:
            return [None] * len(params_list)
        
        if use_parallel and len(params_list) > 1:
            # Use multiprocessing for parallel evaluation
            num_workers = min(cpu_count(), len(params_list))
            worker_inputs = [
                ({"params": p, "index": i}, df_train_all_for_oos, test_df)
                for i, p in enumerate(params_list)
            ]
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(_calculate_oos_accuracy_worker, worker_inputs)
            
            # Sort by index and extract OOS accuracies
            results.sort(key=lambda x: x[0])
            return [acc for _, acc in results]
        else:
            # Sequential evaluation
            return [calculate_oos_accuracy(p) for p in params_list]

    # Initialize population with diverse parameters
    population = []
    params_list = []
    for i in range(population_size):
        p = random_params()
        fitness = evaluate_params(df, cutoff_date, p)
        population.append({"params": p, "fitness": fitness, "oos_accuracy": None})
        params_list.append(p)
    
    # Calculate OOS accuracy in parallel for initial population
    if test_df is not None:
        if verbose:
            print(f"Calculating OOS accuracy for initial population of {population_size} individuals (parallel)...")
        oos_accs = calculate_oos_accuracy_batch(params_list, use_parallel=True)
        for i, oos_acc in enumerate(oos_accs):
            population[i]["oos_accuracy"] = oos_acc

    best_ind = max(population, key=lambda ind: ind["fitness"])
    if verbose:
        fitnesses = [ind["fitness"] for ind in population]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        oos_accs = [ind["oos_accuracy"] for ind in population if ind["oos_accuracy"] is not None]
        oos_str = f", OOS={best_ind['oos_accuracy']:.4f}" if best_ind["oos_accuracy"] is not None else ""
        if oos_accs:
            avg_oos = sum(oos_accs) / len(oos_accs)
            oos_str += f" (avg={avg_oos:.4f})"
        print(
            f"Initial population: best={best_ind['fitness']:.4f}{oos_str}, "
            f"avg={avg_fitness:.4f}, "
            f"params: {best_ind['params']}"
        )

    # GA loop
    for gen in range(generations):
        new_population = []

        # Elitism - keep top individual (OOS already calculated)
        population.sort(key=lambda ind: ind["fitness"], reverse=True)
        elite = population[0].copy()  # Make a copy to avoid reference issues
        new_population.append(elite)

        # Fill the rest
        children_needed = population_size - len(new_population)
        child_params_list = []
        child_fitness_list = []
        
        while len(new_population) < population_size:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            child_params = crossover(parent1, parent2)
            child_params = mutate(child_params)
            
            fitness = evaluate_params(df, cutoff_date, child_params)
            child_params_list.append(child_params)
            child_fitness_list.append(fitness)
            new_population.append({"params": child_params, "fitness": fitness, "oos_accuracy": None})
        
        # Calculate OOS accuracy in parallel for new children
        if test_df is not None and children_needed > 0:
            if verbose:
                print(f"  Calculating OOS for {children_needed} new children in generation {gen + 1} (parallel)...")
            oos_accs = calculate_oos_accuracy_batch(child_params_list, use_parallel=True)
            # Update OOS accuracies (skip the elite which is already at index 0)
            for i, oos_acc in enumerate(oos_accs):
                new_population[i + 1]["oos_accuracy"] = oos_acc

        population = new_population
        gen_best = max(population, key=lambda ind: ind["fitness"])
        if gen_best["fitness"] > best_ind["fitness"]:
            best_ind = gen_best

        if verbose:
            # Calculate population diversity metrics
            fitnesses = [ind["fitness"] for ind in population]
            avg_fitness = sum(fitnesses) / len(fitnesses)
            min_fitness = min(fitnesses)
            max_fitness = max(fitnesses)
            fitness_std = (sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5
            
            # Calculate OOS metrics
            oos_accs = [ind["oos_accuracy"] for ind in population if ind["oos_accuracy"] is not None]
            oos_str = ""
            if oos_accs:
                best_oos = max(oos_accs)
                avg_oos = sum(oos_accs) / len(oos_accs)
                min_oos = min(oos_accs)
                max_oos = max(oos_accs)
                oos_str = f", OOS: best={best_oos:.4f}, avg={avg_oos:.4f}, range=[{min_oos:.4f}, {max_oos:.4f}]"
            
            # Calculate parameter diversity (sample a few params to check)
            sample_k_values = [ind["params"]["k"] for ind in population[:5]]
            k_range = max(sample_k_values) - min(sample_k_values)
            
            print(
                f"Gen {gen + 1:02d}, best={gen_best['fitness']:.4f}, "
                f"avg={avg_fitness:.4f}, std={fitness_std:.4f}, "
                f"range=[{min_fitness:.4f}, {max_fitness:.4f}], "
                f"k_range={k_range:.1f}{oos_str}"
            )
            if gen_best["fitness"] > best_ind["fitness"]:
                oos_note = f", OOS={gen_best['oos_accuracy']:.4f}" if gen_best["oos_accuracy"] is not None else ""
                print(f"  *** NEW BEST: {gen_best['params']}{oos_note}")

        if return_all_results:
            # Store generation summary
            oos_accs = [ind['oos_accuracy'] for ind in population if ind['oos_accuracy'] is not None]
            gen_summary = {
                'generation': gen + 1,
                'best_fitness': gen_best['fitness'],
                'best_params': gen_best['params'].copy(),
                'population_avg_fitness': sum(ind['fitness'] for ind in population) / len(population),
                'population_max_fitness': gen_best['fitness'],
                'population_min_fitness': min(ind['fitness'] for ind in population),
            }
            if oos_accs:
                gen_summary['best_oos_accuracy'] = max(oos_accs)
                gen_summary['avg_oos_accuracy'] = sum(oos_accs) / len(oos_accs)
                gen_summary['min_oos_accuracy'] = min(oos_accs)
                gen_summary['max_oos_accuracy'] = max(oos_accs)
            all_results.append(gen_summary)

    if return_all_results:
        return best_ind["params"], best_ind["fitness"], cutoff_date, all_results
    return best_ind["params"], best_ind["fitness"], cutoff_date


def save_comprehensive_results(output_path, best_params, best_fitness, extended_metrics,
                               all_results=None, config_name="default"):
    """
    Save comprehensive GA results to a JSON file.
    
    This function creates a detailed results file with:
    - Best configuration parameters
    - All metrics (ROI, accuracy, calibration, consistency, etc.)
    - Calibration curve data for plotting
    - Consistency breakdown by tier, method, time
    - Per-generation evolution data
    
    Args:
        output_path: Path to save the JSON file
        best_params: Dict of best GA parameters
        best_fitness: Best fitness value achieved
        extended_metrics: Dict from evaluate_params_roi with return_extended=True
        all_results: Optional list of per-generation results
        config_name: Name identifier for this configuration
    
    Returns:
        dict: The saved results structure
    """
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    # Build calibration curve data for plotting
    calibration_data = None
    comprehensive = extended_metrics.get('comprehensive_metrics')
    if comprehensive:
        bin_data = comprehensive.get('calibration', {}).get('bin_data', [])
        if bin_data:
            calibration_data = {
                'bins': [
                    {
                        'range': f"{int(b['bin_lower']*100)}-{int(b['bin_upper']*100)}%",
                        'predicted': b['avg_predicted'],
                        'actual': b['actual_rate'],
                        'count': b['count']
                    }
                    for b in bin_data
                ]
            }
    
    # Build consistency breakdown
    consistency_breakdown = None
    if comprehensive:
        consistency = comprehensive.get('consistency', {})
        consistency_breakdown = {
            'by_elo_tier': consistency.get('by_elo_tier', {}).get('tiers', {}),
            'by_method': consistency.get('by_method', {}).get('methods', {}),
            'by_time_period': consistency.get('by_time_period', {}).get('periods', {}),
            'by_experience_gap': consistency.get('by_experience_gap', {}).get('gaps', {}),
        }
    
    # Build ROI by decile data
    roi_by_decile = None
    if comprehensive:
        perf = comprehensive.get('performance', {})
        roi_by_decile = perf.get('roi_by_decile', {}).get('deciles', {})
    
    results = {
        'config_name': config_name,
        'timestamp': pd.Timestamp.now().isoformat(),
        
        # Core results
        'best_params': convert_to_serializable(best_params),
        'best_fitness': convert_to_serializable(best_fitness),
        
        # Summary metrics
        'summary': {
            'roi_percent': convert_to_serializable(extended_metrics.get('roi_percent')),
            'accuracy': convert_to_serializable(extended_metrics.get('accuracy')),
            'sharpe_ratio': convert_to_serializable(extended_metrics.get('sharpe_ratio')),
            'win_rate': convert_to_serializable(extended_metrics.get('win_rate')),
            'num_bets': convert_to_serializable(extended_metrics.get('num_bets')),
            # Calibration
            'ece': convert_to_serializable(extended_metrics.get('ece')),
            'brier_score': convert_to_serializable(extended_metrics.get('brier_score')),
            'log_loss': convert_to_serializable(extended_metrics.get('log_loss')),
            'calibration_slope': convert_to_serializable(extended_metrics.get('calibration_slope')),
            # Performance
            'auc_roc': convert_to_serializable(extended_metrics.get('auc_roc')),
            # Consistency
            'consistency_variance': convert_to_serializable(extended_metrics.get('consistency_variance')),
            # Betting
            'max_drawdown': convert_to_serializable(extended_metrics.get('max_drawdown')),
        },
        
        # Detailed calibration curve data for plotting
        'calibration_curve': convert_to_serializable(calibration_data),
        
        # Consistency breakdown by dimension
        'consistency_breakdown': convert_to_serializable(consistency_breakdown),
        
        # ROI by confidence decile
        'roi_by_decile': convert_to_serializable(roi_by_decile),
        
        # Per-generation evolution (if available)
        'evolution': convert_to_serializable(all_results) if all_results else None,
        
        # Full comprehensive metrics (for detailed analysis)
        'comprehensive_metrics': convert_to_serializable(comprehensive) if comprehensive else None,
    }
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def print_comprehensive_summary(extended_metrics):
    """
    Print a human-readable summary of all metrics.
    
    Args:
        extended_metrics: Dict from evaluate_params_roi with return_extended=True
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE METRICS SUMMARY")
    print("="*60)
    
    # Core Metrics
    print("\n--- Core Metrics ---")
    print(f"ROI: {extended_metrics.get('roi_percent', 0):.2f}%")
    print(f"Accuracy: {extended_metrics.get('accuracy', 0)*100:.1f}%")
    print(f"Win Rate: {extended_metrics.get('win_rate', 0)*100:.0f}%")
    print(f"Sharpe Ratio: {extended_metrics.get('sharpe_ratio', 'N/A')}")
    print(f"Trend: {extended_metrics.get('trend', 'N/A')}")
    
    # Calibration Metrics
    print("\n--- Calibration Metrics ---")
    print(f"Expected Calibration Error (ECE): {extended_metrics.get('ece', 'N/A')}")
    print(f"Brier Score: {extended_metrics.get('brier_score', 'N/A')}")
    print(f"Log Loss: {extended_metrics.get('log_loss', 'N/A')}")
    calib_slope = extended_metrics.get('calibration_slope')
    if calib_slope:
        status = "WELL CALIBRATED"
        if calib_slope < 0.9:
            status = "OVERCONFIDENT"
        elif calib_slope > 1.1:
            status = "UNDERCONFIDENT"
        print(f"Calibration Slope: {calib_slope:.2f} ({status})")
    
    # Performance Metrics
    print("\n--- Performance Metrics ---")
    print(f"AUC-ROC: {extended_metrics.get('auc_roc', 'N/A')}")
    
    # Consistency Metrics
    print("\n--- Consistency Metrics ---")
    cons_var = extended_metrics.get('consistency_variance')
    if cons_var is not None:
        status = "CONSISTENT" if cons_var < 0.005 else "HIGH VARIANCE"
        print(f"Consistency Variance: {cons_var:.4f} ({status})")
    
    # Betting Metrics
    print("\n--- Betting Metrics ---")
    print(f"Max Drawdown: ${extended_metrics.get('max_drawdown', 'N/A')}")
    
    # Comprehensive metrics details
    comprehensive = extended_metrics.get('comprehensive_metrics')
    if comprehensive:
        # Calibration curve summary
        bin_data = comprehensive.get('calibration', {}).get('bin_data', [])
        if bin_data:
            print("\n--- Calibration Curve ---")
            for b in bin_data:
                if b['count'] > 0:
                    pred = b['avg_predicted']
                    actual = b['actual_rate'] or 0
                    diff = abs(pred - actual) if b['actual_rate'] else None
                    diff_str = f"(diff: {diff:.2f})" if diff else ""
                    print(f"  {int(b['bin_lower']*100):2d}-{int(b['bin_upper']*100):3d}%: "
                          f"pred={pred:.2f}, actual={actual:.2f}, n={b['count']} {diff_str}")
        
        # High variance flags
        flags = comprehensive.get('consistency', {}).get('high_variance_flags', [])
        if flags:
            print(f"\n⚠️  HIGH VARIANCE FLAGS: {', '.join(flags)}")
        
        # Time trend
        time_data = comprehensive.get('consistency', {}).get('by_time_period', {})
        if time_data.get('is_degrading'):
            print("\n⚠️  WARNING: Accuracy appears to be degrading over time")
    
    print("\n" + "="*60)


# =========================
# Main
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Elo Parameter Optimization")
    parser.add_argument("--mode", choices=["accuracy", "roi"], default="roi",
                        help="Optimization mode: 'accuracy' (original) or 'roi' (ROI-based)")
    parser.add_argument("--population", type=int, default=30, help="Population size")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--lookback-days", type=int, default=365, dest="lookback_days",
                        help="Number of days to look back for ROI optimization (default: 365). "
                             "Use 0 for all available data. Only applies to --mode roi.")
    parser.add_argument("--decay-mode", choices=["linear", "exponential", "none"], default="none",
                        dest="decay_mode",
                        help="Decay mode for Elo ratings: 'linear', 'exponential', or 'none' (default: none). "
                             "When set to 'linear' or 'exponential', decay_rate and min_days will be "
                             "included in GA optimization.")
    
    # New feature flags
    parser.add_argument("--multiphase-decay", choices=["on", "off"], default="off",
                        dest="multiphase_decay",
                        help="Enable multiphase decay feature (default: off). "
                             "Implements piecewise decay with quick succession bump and exponential decay.")
    parser.add_argument("--weight-adjust", choices=["on", "off"], default="off",
                        dest="weight_adjust",
                        help="Enable weight class adjustment feature (default: off). "
                             "Applies penalty for moving up in weight class and bonus for winning.")
    parser.add_argument("--optimize-elo-denom", choices=["on", "off"], default="off",
                        dest="optimize_elo_denom",
                        help="Enable Elo denominator optimization (default: off). "
                             "Makes the '400' constant in Elo equation a GA-optimizable variable.")
    
    # Multi-metric optimization options
    parser.add_argument("--fitness-mode", choices=["roi", "multi"], default="roi",
                        dest="fitness_mode",
                        help="Fitness evaluation mode: 'roi' for ROI-only, 'multi' for weighted multi-metric. "
                             "When 'multi', uses --fitness-weights to combine ROI, calibration, consistency, and AUC.")
    parser.add_argument("--fitness-weights", type=str, default=None,
                        dest="fitness_weights",
                        help="Weights for multi-metric fitness (JSON format). "
                             "Example: '{\"roi\": 0.4, \"calibration\": 0.2, \"consistency\": 0.2, \"auc\": 0.2}'")
    parser.add_argument("--output-json", type=str, default=None,
                        dest="output_json",
                        help="Path to save comprehensive results JSON file.")
    parser.add_argument("--show-comprehensive", action="store_true",
                        dest="show_comprehensive",
                        help="Show comprehensive metrics summary after optimization.")
    
    args = parser.parse_args()
    
    # Convert on/off to boolean
    multiphase_decay = args.multiphase_decay == "on"
    weight_adjust = args.weight_adjust == "on"
    optimize_elo_denom = args.optimize_elo_denom == "on"
    
    # Parse fitness weights if provided
    fitness_weights = None
    if args.fitness_mode == "multi":
        if args.fitness_weights:
            try:
                fitness_weights = json.loads(args.fitness_weights)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse --fitness-weights '{args.fitness_weights}'. Using defaults.")
                fitness_weights = {'roi': 0.4, 'calibration': 0.2, 'consistency': 0.2, 'auc': 0.2}
        else:
            # Default multi-metric weights
            fitness_weights = {'roi': 0.4, 'calibration': 0.2, 'consistency': 0.2, 'auc': 0.2}
    
    df = pd.read_csv("data/interleaved_cleaned.csv", low_memory=False)
    test_df = pd.read_csv("data/past3_events.csv", low_memory=False)
    odds_df = pd.read_csv("after_averaging.csv", low_memory=False)

    df["result"] = pd.to_numeric(df["result"], errors="coerce")
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.tz_localize(None)
    df = df.sort_values("DATE").reset_index(drop=True)
    
    odds_df["DATE"] = pd.to_datetime(odds_df["DATE"]).dt.tz_localize(None)

    # boutcounts first, so prediction filters can use them
    df = add_bout_counts(df)
    # Ensure boutcount columns are numeric (they may be strings from CSV)
    if "precomp_boutcount" in df.columns:
        df["precomp_boutcount"] = pd.to_numeric(df["precomp_boutcount"], errors="coerce")
    if "opp_precomp_boutcount" in df.columns:
        df["opp_precomp_boutcount"] = pd.to_numeric(df["opp_precomp_boutcount"], errors="coerce")

    print(f"Test data date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Running in mode: {args.mode}")
    print(f"Decay mode: {args.decay_mode}")
    
    # Log active feature flags
    print(f"Feature flags: multiphase-decay={args.multiphase_decay}, weight-adjust={args.weight_adjust}, optimize-elo-denom={args.optimize_elo_denom}")

    if args.mode == "roi":
        print("\n" + "="*60)
        print("ROI-BASED GENETIC ALGORITHM OPTIMIZATION")
        print("="*60)
        
        # Calculate and display lookback period details
        max_date = df["DATE"].max()
        if args.lookback_days and args.lookback_days > 0:
            lookback_cutoff = max_date - pd.Timedelta(days=args.lookback_days)
            print(f"Optimizing ROI on fights from {lookback_cutoff.date()} to {max_date.date()} (last {args.lookback_days} days)")
            # Count fights in lookback period
            fights_in_lookback = len(df[df["DATE"] > lookback_cutoff])
            # Count unique fights with odds in lookback period
            odds_df_copy = odds_df.copy()
            odds_in_lookback = odds_df_copy[odds_df_copy["DATE"] > lookback_cutoff]
            odds_fights_in_lookback = len(odds_in_lookback[['DATE', 'FIGHTER', 'opp_FIGHTER']].drop_duplicates()) // 2
            print(f"Fights in lookback period: {fights_in_lookback}")
            print(f"Fights with odds in lookback period: ~{odds_fights_in_lookback}")
        else:
            print(f"Optimizing ROI on ALL fights with available odds")
            odds_min_date = odds_df["DATE"].min()
            odds_max_date = odds_df["DATE"].max()
            print(f"Odds data date range: {odds_min_date.date()} to {odds_max_date.date()}")
        
        # Count total fights with odds for reference
        total_odds_fights = len(odds_df[['DATE', 'FIGHTER', 'opp_FIGHTER']].drop_duplicates()) // 2
        print(f"Total fights with odds data: ~{total_odds_fights}")
        print(f"Total historical fights: {len(df)}")
        if fitness_weights:
            print(f"Fitness mode: multi-metric with weights: {fitness_weights}")
        else:
            print(f"Fitness mode: ROI-only")
        print("Calling ga_search_params_roi()...")
        
        # Run ROI-based GA search with lookback_days parameter and decay_mode
        best_params, best_roi, all_results = ga_search_params_roi(
            df,
            odds_df,
            test_df=test_df,
            population_size=args.population,
            generations=args.generations,
            lookback_days=args.lookback_days,
            seed=args.seed,
            decay_mode=args.decay_mode,
            multiphase_decay=multiphase_decay,
            weight_adjust=weight_adjust,
            optimize_elo_denom=optimize_elo_denom,
            fitness_weights=fitness_weights,
            return_all_results=True,
        )

        print("\n=== GA best params (ROI-optimized) ===")
        print(best_params)
        if args.lookback_days and args.lookback_days > 0:
            print(f"Best ROI on fights in last {args.lookback_days} days: {best_roi:.2f}%")
        else:
            print(f"Best ROI on all fights with odds: {best_roi:.2f}%")

        # Train final Elo with best params on ALL DATA before test dates
        mov_params = {
            "w_ko":   best_params["w_ko"],
            "w_sub":  best_params["w_sub"],
            "w_udec": best_params["w_udec"],
            "w_sdec": best_params["w_sdec"],
            "w_mdec": best_params["w_mdec"],
        }
        best_k = best_params["k"]
        best_decay_rate = best_params.get("decay_rate", 0.0)
        best_min_days = best_params.get("min_days", 180)
        
        # Get elo_denom from best params if optimize_elo_denom was enabled
        best_elo_denom = best_params.get("elo_denom", 400) if optimize_elo_denom else 400
        
        # Prepare multiphase decay params if feature was enabled
        best_multiphase_decay_params = None
        if multiphase_decay:
            best_multiphase_decay_params = {
                "quick_succession_days": best_params.get("quick_succession_days", 30),
                "quick_succession_bump": best_params.get("quick_succession_bump", 1.05),
                "decay_days": best_params.get("decay_days", 180),
                "multiphase_decay_rate": best_params.get("multiphase_decay_rate", 0.002),
            }
        
        # Prepare weight adjust params if feature was enabled
        best_weight_adjust_params = None
        weight_history = None
        if weight_adjust:
            best_weight_adjust_params = {
                "weight_up_precomp_penalty": best_params.get("weight_up_precomp_penalty", 0.95),
                "weight_up_postcomp_bonus": best_params.get("weight_up_postcomp_bonus", 1.10),
            }
            # Build weight history for the training data
            weight_history = build_fighter_weight_history(df)

        # Use ALL data before the test dates for final training
        test_start_date = pd.to_datetime(test_df["date"]).min()
        df_train_all = df[df["DATE"] < test_start_date].copy()
        df_trained = run_basic_elo(
            df_train_all, k=best_k, mov_params=mov_params, denominator=best_elo_denom,
            decay_mode=args.decay_mode, decay_rate=best_decay_rate, min_days=best_min_days,
            multiphase_decay_params=best_multiphase_decay_params,
            weight_adjust_params=best_weight_adjust_params,
            weight_history=weight_history
        )

        # OOS ROI evaluation on past 3 events
        # Use test_df as the odds source for OOS evaluation (it has avg_odds column)
        test_odds_df = test_df.copy()
        test_odds_df = test_odds_df.rename(columns={
            'date': 'DATE',
            'fighter': 'FIGHTER',
            'opp_fighter': 'opp_FIGHTER'
        })
        test_odds_df["DATE"] = pd.to_datetime(test_odds_df["DATE"]).dt.tz_localize(None)
        
        print("\n=== OOS ROI evaluation on data/past3_events.csv ===")
        oos_roi_results = calculate_oos_roi(
            df_trained,
            test_df,
            test_odds_df,
            verbose=True,
        )
        print(f"\nOOS Results:")
        print(f"  Total Bets: {oos_roi_results['total_bets']}")
        print(f"  Wins: {oos_roi_results['wins']}")
        print(f"  Accuracy: {oos_roi_results['accuracy']*100:.2f}%")
        print(f"  Total Wagered: ${oos_roi_results['total_wagered']:.2f}")
        print(f"  Total Profit: ${oos_roi_results['total_profit']:.2f}")
        print(f"  ROI: {oos_roi_results['roi_percent']:.2f}%")
        
        # Also calculate OOS accuracy for comparison
        print("\n=== OOS Accuracy evaluation on data/past3_events.csv ===")
        oos_acc = test_out_of_sample_accuracy(
            df_trained,
            test_df,
            verbose=True,
            gap_threshold=75,
            min_train_bouts=1,
        )
        print("Overall OOS accuracy:", oos_acc)
        
        # Compute comprehensive metrics on final trained model
        print("\n=== Computing Comprehensive Metrics ===")
        final_extended = evaluate_params_roi(
            df, odds_df, best_params, lookback_days=args.lookback_days,
            return_extended=True, decay_mode=args.decay_mode,
            multiphase_decay=multiphase_decay, weight_adjust=weight_adjust,
            optimize_elo_denom=optimize_elo_denom, weight_history=weight_history
        )
        
        # Show comprehensive summary if requested
        if args.show_comprehensive:
            print_comprehensive_summary(final_extended)
        
        # Save to JSON if output path provided
        if args.output_json:
            print(f"\nSaving comprehensive results to: {args.output_json}")
            save_comprehensive_results(
                args.output_json,
                best_params,
                best_roi,
                final_extended,
                all_results,
                config_name=f"ga_roi_{args.lookback_days}d"
            )
            print(f"Results saved successfully!")
    else:
        # Original accuracy-based GA
        print("\n" + "="*60)
        print("ACCURACY-BASED GENETIC ALGORITHM OPTIMIZATION")
        print("="*60)
        print("Calling ga_search_params()...")
        
        # Run GA search over k and MoV weights
        best_params, best_future_acc, cutoff = ga_search_params(
            df,
            test_df=test_df,
            population_size=args.population,
            generations=args.generations,
            cutoff_quantile=0.8,
            seed=args.seed,
        )

        print("\n=== GA best params ===")
        print(best_params)
        print(f"Best future accuracy on training window: {best_future_acc:.4f}")

        # Train final Elo with best params on ALL DATA before test dates
        mov_params = {
            "w_ko":   best_params["w_ko"],
            "w_sub":  best_params["w_sub"],
            "w_udec": best_params["w_udec"],
            "w_sdec": best_params["w_sdec"],
            "w_mdec": best_params["w_mdec"],
        }
        best_k = best_params["k"]

        # Use ALL data before the test dates for final training
        test_start_date = pd.to_datetime(test_df["date"]).min()
        df_train_all = df[df["DATE"] < test_start_date].copy()
        df_trained = run_basic_elo(df_train_all, k=best_k, mov_params=mov_params)

        # OOS evaluation on past 3 events
        print("\n=== OOS evaluation on data/past3_events.csv ===")
        oos_acc = test_out_of_sample_accuracy(
            df_trained,
            test_df,
            verbose=True,
            gap_threshold=75,
            min_train_bouts=1,
        )
        print("Overall OOS accuracy:", oos_acc)

