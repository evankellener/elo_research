"""
Shared utilities for Elo rating system calculations.
Contains common functions used across multiple scripts.
"""
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

