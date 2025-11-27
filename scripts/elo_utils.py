"""
Shared utilities for Elo rating system calculations.
Contains common functions used across multiple scripts.
"""
import math
import pandas as pd


def is_true(val):
    """
    Helper to safely check if a value indicates the event occurred.
    Handles both string and numeric values for MOV flags.
    """
    if pd.isna(val):
        return False
    # Handle string values - check if it's '1' or the column name itself
    if isinstance(val, str):
        return val.strip() == '1' or val.strip().lower() in ['true', '1']
    # Handle numeric values
    return val == 1 or val == True


def method_of_victory_scale(row,
                            w_ko=1.4,
                            w_sub=1.3,
                            w_udec=1.0,
                            w_mdec=0.9,
                            w_sdec=0.7):
    """
    Return a multiplier for K based on method of victory.

    We look at the row from the FIGHTER perspective, but the flags
    (ko, kod, subw, subwd, etc) encode outcome for both sides.
    For the Elo update we just care about how decisive the fight was,
    not who got it, so we collapse to categories.
    
    Note: Handles both string and numeric values for MOV flags.
    
    Args:
        row: DataFrame row containing MOV flags
        w_ko: Weight for KO/TKO (default 1.4)
        w_sub: Weight for submission (default 1.3)
        w_udec: Weight for unanimous decision (default 1.0)
        w_mdec: Weight for majority decision (default 0.9)
        w_sdec: Weight for split decision (default 0.7)
    
    Returns:
        Multiplier value for K-factor
    """
    # KO or TKO
    if is_true(row.get("ko")) or is_true(row.get("kod")):
        return w_ko
    # Submission
    if is_true(row.get("subw")) or is_true(row.get("subwd")):
        return w_sub
    # Unanimous decision
    if is_true(row.get("udec")) or is_true(row.get("udecd")):
        return w_udec
    # Majority decision
    if is_true(row.get("mdec")) or is_true(row.get("mdecd")):
        return w_mdec
    # Split decision
    if is_true(row.get("sdec")) or is_true(row.get("sdecd")):
        return w_sdec
    # Fallback when we do not recognize the type
    return 1.0


def mov_factor(row, params):
    """
    Compute method of victory multiplier for this fight using a params dict.
    This is a wrapper around method_of_victory_scale for use with genetic algorithms.
    
    Args:
        row: DataFrame row containing MOV flags
        params: Dict with keys w_ko, w_sub, w_udec, w_sdec, w_mdec
    
    Returns:
        Multiplier value for K-factor
    """
    return method_of_victory_scale(
        row,
        w_ko=params.get("w_ko", 1.2316676566177203),#1.4
        w_sub=params.get("w_sub", 1.762222415337883),#1.3
        w_udec=params.get("w_udec", 1.0982822750786363),#1.0
        w_mdec=params.get("w_mdec", 1.081378052586788),#0.9
        w_sdec=params.get("w_sdec", 0.5129713578039266),#0.7
    )


def build_fighter_history(df):
    """
    Build a unified history DataFrame from both FIGHTER and opp_FIGHTER perspectives.
    
    Args:
        df: DataFrame with columns DATE, FIGHTER, opp_FIGHTER, precomp_elo, 
            postcomp_elo, opp_precomp_elo, opp_postcomp_elo
    
    Returns:
        DataFrame with columns date, fighter, pre_elo, post_elo, sorted by date
    """
    a = df[["DATE", "FIGHTER", "precomp_elo", "postcomp_elo"]].rename(
        columns={
            "FIGHTER": "fighter",
            "precomp_elo": "pre_elo",
            "postcomp_elo": "post_elo",
        }
    )
    b = df[["DATE", "opp_FIGHTER", "opp_precomp_elo", "opp_postcomp_elo"]].rename(
        columns={
            "opp_FIGHTER": "fighter",
            "opp_precomp_elo": "pre_elo",
            "opp_postcomp_elo": "post_elo",
        }
    )
    hist = pd.concat([a, b], ignore_index=True)
    return (
        hist.sort_values("DATE")
        .rename(columns={"DATE": "date"})
        .reset_index(drop=True)
    )


def has_prior_history(first_dates, fighter, date):
    """
    Check if a fighter has at least one fight before the given date.
    
    Args:
        first_dates: Dict mapping fighter names to their first fight date
        fighter: Fighter name to check
        date: Date to check against
    
    Returns:
        True if fighter has at least one prior fight, False otherwise
    """
    d0 = first_dates.get(fighter)
    return bool(d0 and date > d0)


def add_bout_counts(df):
    """
    Add bout count columns to the DataFrame if they don't already exist.
    Each fighter's bout count represents how many fights they had BEFORE this bout.
    
    Args:
        df: DataFrame with columns DATE, FIGHTER, opp_FIGHTER (must be sorted by DATE)
    
    Returns:
        DataFrame with columns precomp_boutcount and opp_precomp_boutcount
        (only adds them if they don't already exist)
    """
    # Check if columns already exist
    if "precomp_boutcount" in df.columns and "opp_precomp_boutcount" in df.columns:
        return df
    
    df = df.sort_values("DATE").copy()
    bout_counts = {}
    pre_counts = []
    opp_pre_counts = []

    for _, r in df.iterrows():
        f1 = r["FIGHTER"]
        f2 = r["opp_FIGHTER"]

        c1 = bout_counts.get(f1, 0)
        c2 = bout_counts.get(f2, 0)

        pre_counts.append(c1)
        opp_pre_counts.append(c2)

        bout_counts[f1] = c1 + 1
        bout_counts[f2] = c2 + 1

    df["precomp_boutcount"] = pre_counts
    df["opp_precomp_boutcount"] = opp_pre_counts
    return df


def build_fighter_last_fight_date(df):
    """
    Build a dictionary mapping fighter names to their last fight date.
    
    This function processes the DataFrame chronologically and tracks the most
    recent fight date for each fighter. It handles both the FIGHTER and
    opp_FIGHTER columns to capture all fights.
    
    Args:
        df: DataFrame with columns DATE, FIGHTER, opp_FIGHTER (must be sorted by DATE)
    
    Returns:
        dict: Mapping of fighter name to their last fight date as pd.Timestamp
    """
    df = df.sort_values("DATE").copy()
    last_fight_dates = {}
    
    for _, row in df.iterrows():
        date = row["DATE"]
        f1 = row["FIGHTER"]
        f2 = row["opp_FIGHTER"]
        
        # Update last fight date for both fighters
        last_fight_dates[f1] = date
        last_fight_dates[f2] = date
    
    return last_fight_dates


def get_last_fight_date_before(fighter, current_date, last_fight_dates_history):
    """
    Get the last fight date for a fighter before a given date.
    
    This is used during Elo calculation to get the last fight date
    as of the current fight being processed. The caller is responsible
    for ensuring the returned date is actually before current_date
    when building the history incrementally.
    
    Args:
        fighter: Fighter name
        current_date: The date of the current fight (for documentation purposes)
        last_fight_dates_history: Dict mapping fighter names to their last fight date
    
    Returns:
        pd.Timestamp or None: The last fight date for the fighter, or None if no prior fight
    """
    return last_fight_dates_history.get(fighter)


def apply_linear_decay(elo, days_since_fight, decay_rate, min_days=180):
    """
    Apply linear decay to an Elo rating based on time since last fight.
    
    Formula: adjusted_elo = elo * (1 - decay_rate * (days_since_fight - min_days))
    
    Only applies decay if days_since_fight >= min_days. The decay is applied
    only to the days beyond min_days, not the total days since last fight.
    The decay is clamped to ensure the Elo doesn't go below 50% of original.
    
    Args:
        elo: Current Elo rating
        days_since_fight: Number of days since fighter's last fight
        decay_rate: Decay rate per day (e.g., 0.0005 = 0.05% per day)
        min_days: Minimum days before decay starts (default 180)
    
    Returns:
        float: Adjusted Elo rating after decay
    """
    if days_since_fight is None or days_since_fight < min_days:
        return elo
    
    # Apply decay only to the days beyond min_days
    effective_days = days_since_fight - min_days
    decay_factor = 1.0 - decay_rate * effective_days
    
    # Clamp to prevent excessive decay (minimum 50% of original)
    decay_factor = max(0.5, decay_factor)
    
    return elo * decay_factor


def apply_exponential_decay(elo, days_since_fight, decay_rate, min_days=180):
    """
    Apply exponential decay to an Elo rating based on time since last fight.
    
    Formula: adjusted_elo = elo * exp(-decay_rate * (days_since_fight - min_days))
    
    Only applies decay if days_since_fight >= min_days. The decay is applied
    only to the days beyond min_days, not the total days since last fight.
    The decay is clamped to ensure the Elo doesn't go below 50% of original.
    
    Args:
        elo: Current Elo rating
        days_since_fight: Number of days since fighter's last fight
        decay_rate: Decay rate per day (e.g., 0.001 = 0.1% per day)
        min_days: Minimum days before decay starts (default 180)
    
    Returns:
        float: Adjusted Elo rating after decay
    """
    if days_since_fight is None or days_since_fight < min_days:
        return elo
    
    # Apply decay only to the days beyond min_days
    effective_days = days_since_fight - min_days
    decay_factor = math.exp(-decay_rate * effective_days)
    
    # Clamp to prevent excessive decay (minimum 50% of original)
    decay_factor = max(0.5, decay_factor)
    
    return elo * decay_factor


def apply_decay(elo, days_since_fight, decay_rate, min_days=180, decay_mode="none"):
    """
    Apply decay to an Elo rating based on the specified decay mode.
    
    Args:
        elo: Current Elo rating
        days_since_fight: Number of days since fighter's last fight
        decay_rate: Decay rate per day
        min_days: Minimum days before decay starts (default 180)
        decay_mode: "linear", "exponential", or "none" (default "none")
    
    Returns:
        float: Adjusted Elo rating after decay (or original if decay_mode is "none")
    """
    if decay_mode == "linear":
        return apply_linear_decay(elo, days_since_fight, decay_rate, min_days)
    elif decay_mode == "exponential":
        return apply_exponential_decay(elo, days_since_fight, decay_rate, min_days)
    else:
        return elo


def apply_multiphase_decay(elo, days_since_fight, quick_succession_days, quick_succession_bump,
                            decay_days, decay_rate):
    """
    Apply multiphase decay adjustment to an Elo rating.
    
    Implements a piecewise function with two phases:
    - Phase 1 (Quick Succession): If days_since_last_fight < quick_succession_days,
      apply quick_succession_bump (positive multiplier > 1.0)
    - Phase 2 (Decay): If days_since_last_fight > decay_days,
      apply exponential decay penalty
    - Between: No adjustment applied
    
    Args:
        elo: Current Elo rating
        days_since_fight: Number of days since fighter's last fight (None for debut)
        quick_succession_days: Days threshold for quick succession bump
        quick_succession_bump: Multiplier for recent fighters (> 1.0 = boost)
        decay_days: Days threshold for decay to start
        decay_rate: Exponential decay rate
    
    Returns:
        float: Adjusted Elo rating
    """
    if days_since_fight is None:
        return elo
    
    if days_since_fight < quick_succession_days:
        # Phase 1: Quick succession bump
        return elo * quick_succession_bump
    elif days_since_fight > decay_days:
        # Phase 2: Exponential decay
        effective_days = days_since_fight - decay_days
        decay_factor = math.exp(-decay_rate * effective_days)
        # Clamp to prevent excessive decay (minimum 50% of original)
        decay_factor = max(0.5, decay_factor)
        return elo * decay_factor
    else:
        # Between: No adjustment
        return elo


def build_fighter_weight_history(df):
    """
    Build a dictionary tracking each fighter's weight class history over time.
    
    This function processes the DataFrame chronologically and builds a history
    of weight classes for each fighter based on their fight weights.
    
    Args:
        df: DataFrame with columns DATE, FIGHTER, opp_FIGHTER, weight_stat (or weight_of_fight)
            Must be sorted by DATE
    
    Returns:
        dict: Mapping of fighter name to list of (date, weight) tuples sorted by date
    """
    df = df.sort_values("DATE").copy()
    weight_history = {}
    
    for _, row in df.iterrows():
        date = row["DATE"]
        f1 = row["FIGHTER"]
        f2 = row["opp_FIGHTER"]
        
        # Get weight for fighter 1
        w1 = row.get("weight_stat")
        if pd.isna(w1):
            w1 = row.get("weight_of_fight")
        
        # Get weight for fighter 2 (from opp_weight_stat or estimate from fight weight)
        w2 = row.get("opp_weight_stat")
        if pd.isna(w2):
            w2 = row.get("weight_of_fight")  # Same fight = same weight class
        
        # Convert to numeric if needed
        if w1 is not None and not pd.isna(w1):
            try:
                w1 = float(w1)
                if f1 not in weight_history:
                    weight_history[f1] = []
                weight_history[f1].append((date, w1))
            except (ValueError, TypeError):
                pass
        
        if w2 is not None and not pd.isna(w2):
            try:
                w2 = float(w2)
                if f2 not in weight_history:
                    weight_history[f2] = []
                weight_history[f2].append((date, w2))
            except (ValueError, TypeError):
                pass
    
    return weight_history


def detect_weight_change(fighter, current_date, current_weight, weight_history):
    """
    Detect if a fighter has moved weight classes.
    
    Args:
        fighter: Fighter name
        current_date: Date of current fight
        current_weight: Weight class of current fight
        weight_history: Dict from build_fighter_weight_history()
    
    Returns:
        tuple: (moved_up, moved_down, previous_weight)
            moved_up: True if fighter moved to higher weight class
            moved_down: True if fighter moved to lower weight class
            previous_weight: The previous weight class (or None if debut)
    """
    if fighter not in weight_history:
        return False, False, None
    
    history = weight_history[fighter]
    
    # Find the most recent fight before current_date
    previous_weight = None
    for date, weight in reversed(history):
        if date < current_date:
            previous_weight = weight
            break
    
    if previous_weight is None:
        return False, False, None
    
    # Compare weights (higher number = heavier weight class)
    if current_weight is None:
        return False, False, previous_weight
    
    try:
        current_w = float(current_weight)
        prev_w = float(previous_weight)
        
        moved_up = current_w > prev_w
        moved_down = current_w < prev_w
        
        return moved_up, moved_down, previous_weight
    except (ValueError, TypeError):
        return False, False, previous_weight


def calculate_expected_value(rating1, rating2, elo_denom=400):
    """
    Calculate expected score for a player with rating1 against rating2.
    
    Standard Elo formula: E1 = 1 / (1 + 10^((R2 - R1) / elo_denom))
    
    The elo_denom parameter (default 400) controls the sensitivity of the
    expected value to rating differences. Lower values make ratings more
    sensitive to differences.
    
    Args:
        rating1: Player 1's rating
        rating2: Player 2's rating
        elo_denom: Denominator in the Elo formula (default 400)
    
    Returns:
        float: Expected score for player 1 (between 0 and 1)
    """
    return 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / elo_denom))

