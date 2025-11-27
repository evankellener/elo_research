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
    Build a dictionary mapping each fighter to their last fight date.
    
    This function tracks the most recent fight date for each fighter chronologically.
    The dictionary is computed once at startup and cached in memory.
    
    Args:
        df: DataFrame with columns DATE, FIGHTER, opp_FIGHTER (must be sorted by DATE)
    
    Returns:
        dict: Mapping fighter name -> last fight date (as pandas Timestamp)
    
    Example:
        >>> last_dates = build_fighter_last_fight_date(df)
        >>> last_dates.get("Conor McGregor")
        Timestamp('2021-07-10 00:00:00')
    """
    df = df.sort_values("DATE").copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    last_fight_dates = {}
    
    for _, r in df.iterrows():
        f1 = r["FIGHTER"]
        f2 = r["opp_FIGHTER"]
        fight_date = r["DATE"]
        
        # Update last fight date for both fighters
        last_fight_dates[f1] = fight_date
        last_fight_dates[f2] = fight_date
    
    return last_fight_dates


def compute_days_since_last_fight(fighter, current_date, last_fight_dates):
    """
    Compute the number of days since a fighter's last fight.
    
    Args:
        fighter: Fighter name
        current_date: Current fight date (pandas Timestamp or datetime)
        last_fight_dates: Dict mapping fighter name -> last fight date
    
    Returns:
        int: Number of days since last fight, or None if no prior fight
    """
    last_date = last_fight_dates.get(fighter)
    if last_date is None:
        return None
    
    # Convert to datetime if needed
    current_date = pd.to_datetime(current_date)
    last_date = pd.to_datetime(last_date)
    
    # Calculate days difference
    days_diff = (current_date - last_date).days
    
    # Should be positive (current date should be after last fight)
    if days_diff <= 0:
        return None
    
    return days_diff


def apply_linear_decay(precomp_elo, days_since_fight, decay_rate, min_days=180, base_elo=1500):
    """
    Apply linear decay to Elo rating based on time since last fight.
    
    Linear decay formula: adjusted_elo = precomp_elo * (1 - decay_rate * effective_days)
    where effective_days = days_since_fight - min_days
    
    The decay only applies if days_since_fight >= min_days.
    The adjusted Elo cannot go below base_elo to prevent unreasonable ratings.
    
    Args:
        precomp_elo: Pre-computation Elo rating
        days_since_fight: Number of days since fighter's last fight
        decay_rate: Decay rate per day (e.g., 0.0005 means 0.05% decay per day)
        min_days: Minimum days before decay starts (default: 180)
        base_elo: Minimum Elo rating (default: 1500)
    
    Returns:
        float: Adjusted Elo rating after decay
    """
    if days_since_fight is None or days_since_fight < min_days:
        return precomp_elo
    
    # Calculate effective decay days (days beyond min_days)
    effective_days = days_since_fight - min_days
    
    # Apply linear decay
    decay_factor = 1 - (decay_rate * effective_days)
    
    # Ensure decay factor doesn't go below a reasonable minimum (e.g., 0.5)
    decay_factor = max(decay_factor, 0.5)
    
    adjusted_elo = precomp_elo * decay_factor
    
    # Don't let Elo go below base_elo
    return max(adjusted_elo, base_elo)


def apply_exponential_decay(precomp_elo, days_since_fight, decay_rate, min_days=180, base_elo=1500):
    """
    Apply exponential decay to Elo rating based on time since last fight.
    
    Exponential decay formula: adjusted_elo = precomp_elo * exp(-decay_rate * effective_days)
    where effective_days = days_since_fight - min_days
    
    The decay only applies if days_since_fight >= min_days.
    The adjusted Elo cannot go below base_elo to prevent unreasonable ratings.
    
    Args:
        precomp_elo: Pre-computation Elo rating
        days_since_fight: Number of days since fighter's last fight
        decay_rate: Decay rate per day (e.g., 0.001 means 0.1% decay per day)
        min_days: Minimum days before decay starts (default: 180)
        base_elo: Minimum Elo rating (default: 1500)
    
    Returns:
        float: Adjusted Elo rating after decay
    """
    if days_since_fight is None or days_since_fight < min_days:
        return precomp_elo
    
    # Calculate effective decay days (days beyond min_days)
    effective_days = days_since_fight - min_days
    
    # Apply exponential decay
    decay_factor = math.exp(-decay_rate * effective_days)
    
    # Ensure decay factor doesn't go below a reasonable minimum (e.g., 0.5)
    decay_factor = max(decay_factor, 0.5)
    
    adjusted_elo = precomp_elo * decay_factor
    
    # Don't let Elo go below base_elo
    return max(adjusted_elo, base_elo)


def apply_decay(precomp_elo, days_since_fight, decay_rate, min_days=180, 
                decay_mode="none", base_elo=1500):
    """
    Apply decay to Elo rating based on time since last fight.
    
    This is a unified interface that supports both linear and exponential decay,
    switchable via decay_mode parameter.
    
    Args:
        precomp_elo: Pre-computation Elo rating
        days_since_fight: Number of days since fighter's last fight (or None for debutants)
        decay_rate: Decay rate per day
        min_days: Minimum days before decay starts (default: 180)
        decay_mode: "linear", "exponential", or "none" (default: "none")
        base_elo: Minimum Elo rating (default: 1500)
    
    Returns:
        float: Adjusted Elo rating after decay (unchanged if decay_mode is "none")
    """
    if decay_mode == "none" or decay_mode is None:
        return precomp_elo
    
    if decay_mode == "linear":
        return apply_linear_decay(precomp_elo, days_since_fight, decay_rate, min_days, base_elo)
    
    if decay_mode == "exponential":
        return apply_exponential_decay(precomp_elo, days_since_fight, decay_rate, min_days, base_elo)
    
    # Unknown mode, return unchanged
    return precomp_elo

