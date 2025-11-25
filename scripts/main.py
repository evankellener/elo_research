import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from elo_utils import method_of_victory_scale, build_fighter_history, has_prior_history, is_true

def run_basic_elo_with_mov(df, k = 167.19618191211478, base_elo = 1500, denominator = 400, draw_k_factor = 0.5, 
                           w_ko=None, w_sub=None, w_udec=None, w_sdec=None, w_mdec=None):
    """
    Run Elo with method of victory scaling.
    
    Args:
        df: DataFrame with fight data
        k: Base K-factor
        base_elo: Starting Elo rating
        denominator: Elo denominator (default 400)
        draw_k_factor: Multiplier for K-factor in draws (default 0.5, meaning draws have half the impact)
        w_ko, w_sub, w_udec, w_sdec, w_mdec: Custom MOV weights (if None, uses defaults from method_of_victory_scale)
    """
    ratings = {}
    pre_elo, post_elos = [], []
    opp_pre_elos, opp_post_elos = [], []

    for _, row in df.iterrows():
        f1, f2, result = row['FIGHTER'], row['opp_FIGHTER'], row['result']
        
        # Check if this is a draw (both win and loss are 0)
        is_draw = (row.get('win', 1) == 0) and (row.get('loss', 1) == 0)
        
        # For draws, use 0.5 as the result (half win for each fighter)
        if is_draw:
            result = 0.5

        ratings.setdefault(f1, base_elo)
        ratings.setdefault(f2, base_elo)

        f1_pre, f2_pre = ratings[f1], ratings[f2]

        expected_f1 = 1/(1+10.0**((f2_pre-f1_pre)/denominator))
        expected_f2 = 1/(1+10.0**((f1_pre-f2_pre)/denominator))

        # Use custom weights if provided, otherwise use defaults
        if w_ko is not None or w_sub is not None or w_udec is not None or w_sdec is not None or w_mdec is not None:
            mov_scale = method_of_victory_scale(row, 
                                               w_ko=w_ko if w_ko is not None else 1.4,
                                               w_sub=w_sub if w_sub is not None else 1.3,
                                               w_udec=w_udec if w_udec is not None else 1.0,
                                               w_sdec=w_sdec if w_sdec is not None else 0.7,
                                               w_mdec=w_mdec if w_mdec is not None else 0.9)
        else:
            mov_scale = method_of_victory_scale(row)
        k_eff = k * mov_scale
        
        # Reduce K-factor for draws (draws are less decisive)
        if is_draw:
            k_eff = k_eff * draw_k_factor

        new_f1_rating = f1_pre + k_eff * (result - expected_f1)
        new_f2_rating = f2_pre + k_eff * ((1 - result) - expected_f2)

        ratings[f1] = new_f1_rating
        ratings[f2] = new_f2_rating

        pre_elo.append(f1_pre)
        post_elos.append(new_f1_rating)
        opp_pre_elos.append(f2_pre)
        opp_post_elos.append(new_f2_rating)

    df['precomp_elo'] = pre_elo
    df['postcomp_elo'] = post_elos
    df['opp_precomp_elo'] = opp_pre_elos
    df['opp_postcomp_elo'] = opp_post_elos

    return df

def run_basic_elo(df, k = 32, base_elo = 1500, denominator = 400, draw_k_factor = 0.5):
    """
    Run basic Elo rating system.
    
    Args:
        df: DataFrame with fight data
        k: Base K-factor
        base_elo: Starting Elo rating
        denominator: Elo denominator (default 400)
        draw_k_factor: Multiplier for K-factor in draws (default 0.5, meaning draws have half the impact)
    """
    ratings = {}
    pre_elo, post_elos = [], []
    opp_pre_elos, opp_post_elos = [], []

    for _, row in df.iterrows():
        f1, f2, result = row['FIGHTER'], row['opp_FIGHTER'], row['result']
        
        # Check if this is a draw (both win and loss are 0)
        is_draw = (row.get('win', 1) == 0) and (row.get('loss', 1) == 0)
        
        # For draws, use 0.5 as the result (half win for each fighter)
        if is_draw:
            result = 0.5

        ratings.setdefault(f1, base_elo)
        ratings.setdefault(f2, base_elo)

        f1_pre, f2_pre = ratings[f1], ratings[f2]

        expected_f1 = 1/(1+10.0**((f2_pre-f1_pre)/denominator))
        expected_f2 = 1/(1+10.0**((f1_pre-f2_pre)/denominator))

        # Reduce K-factor for draws (draws are less decisive)
        k_eff = k * (draw_k_factor if is_draw else 1.0)

        new_f1_rating = f1_pre + k_eff * (result - expected_f1)
        new_f2_rating = f2_pre + k_eff * ((1 - result) - expected_f2)

        ratings[f1] = new_f1_rating
        ratings[f2] = new_f2_rating

        pre_elo.append(f1_pre)
        post_elos.append(new_f1_rating)
        opp_pre_elos.append(f2_pre)
        opp_post_elos.append(new_f2_rating)

    df['precomp_elo'] = pre_elo
    df['postcomp_elo'] = post_elos
    df['opp_precomp_elo'] = opp_pre_elos
    df['opp_postcomp_elo'] = opp_post_elos

    return df


def compute_fight_predictions(df):
    history = build_fighter_history(df)
    first_fight_dates = history.groupby('fighter')['date'].min().to_dict()
    records = []
    for _, row in df.iterrows():
        if row['result'] not in (0, 1):
            continue
        if pd.isna(row['DATE']):
            continue
        if row['precomp_elo'] == row['opp_precomp_elo']:
            continue
        if not has_prior_history(first_fight_dates, row['FIGHTER'], row['DATE']):
            continue
        if not has_prior_history(first_fight_dates, row['opp_FIGHTER'], row['DATE']):
            continue
        predicted = 1 if row['precomp_elo'] > row['opp_precomp_elo'] else 0
        records.append(
            {
                'date': row['DATE'],
                'prediction': predicted,
                'result': int(row['result']),
                'correct': int(predicted == row['result']),
            }
        )
    return pd.DataFrame(records)


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


def compute_roi_predictions(df, odds_df=None):
    """
    Calculate ROI, log loss, and brier score for Elo-based betting predictions.
    
    This function simulates betting on the fighter with the higher Elo rating
    and calculates returns based on betting odds.
    
    Args:
        df: DataFrame with fight data including precomp_elo, opp_precomp_elo, 
            result, and DATE columns (Elo values calculated by run_basic_elo)
        odds_df: Optional DataFrame with odds data (e.g., from after_averaging.csv).
                 If provided, odds are merged by matching FIGHTER, opp_FIGHTER, and DATE.
                 If not provided, df must contain avg_odds column.
    
    Returns:
        dict: Dictionary containing ROI%, log loss, brier score, and detailed records
    """
    # If odds_df is provided, merge odds into df
    odds_lookup = {}  # Will be used for efficient lookup when betting on opponent
    if odds_df is not None:
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Prepare odds lookup from odds_df
        odds_df = odds_df.copy()
        # Normalize dates - remove timezone info for consistent matching
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        
        # Create a lookup dictionary for odds: (fighter, opponent, date) -> avg_odds
        # This contains both perspectives from odds_df (both fighters' odds)
        for _, row in odds_df.iterrows():
            if pd.notna(row.get('avg_odds')):
                key = (row['FIGHTER'], row['opp_FIGHTER'], str(row['DATE']))
                odds_lookup[key] = row['avg_odds']
        
        # Add odds to df by matching
        def get_odds(row):
            key = (row['FIGHTER'], row['opp_FIGHTER'], str(row['DATE']))
            return odds_lookup.get(key, np.nan)
        
        df['avg_odds'] = df.apply(get_odds, axis=1)
        print(f"Matched {df['avg_odds'].notna().sum()} fights with odds data out of {len(df)} total fights")
    
    history = build_fighter_history(df)
    first_fight_dates = history.groupby('fighter')['date'].min().to_dict()
    
    records = []
    total_wagered = 0
    total_returned = 0
    processed_fights = set()  # Track unique fights to avoid double counting
    
    # Build odds lookup for efficient access when betting on opponent
    # Use odds_lookup if it was built from odds_df (contains both perspectives),
    # otherwise build from df with bidirectional storage
    if odds_lookup:
        # odds_lookup from odds_df already has both perspectives (both fighters' correct odds)
        odds_by_fight = odds_lookup
    else:
        # Build from df - store both directions so lookup works regardless of which fighter has higher Elo
        odds_by_fight = {}
        for _, row in df.iterrows():
            if pd.notna(row.get('avg_odds')):
                key = (row['FIGHTER'], row['opp_FIGHTER'], str(row['DATE']))
                odds_by_fight[key] = row['avg_odds']
                # Also store reverse key so lookup works when opponent has higher Elo
                reverse_key = (row['opp_FIGHTER'], row['FIGHTER'], str(row['DATE']))
                if reverse_key not in odds_by_fight:
                    odds_by_fight[reverse_key] = row['avg_odds']
    
    for _, row in df.iterrows():
        # Skip invalid results
        if row['result'] not in (0, 1):
            continue
        if pd.isna(row['DATE']):
            continue
        if row['precomp_elo'] == row['opp_precomp_elo']:
            continue
        if not has_prior_history(first_fight_dates, row['FIGHTER'], row['DATE']):
            continue
        if not has_prior_history(first_fight_dates, row['opp_FIGHTER'], row['DATE']):
            continue
        
        # Create a unique key for each fight (sorted fighter names + date)
        fight_key = tuple(sorted([row['FIGHTER'], row['opp_FIGHTER']])) + (str(row['DATE']),)
        if fight_key in processed_fights:
            continue
        processed_fights.add(fight_key)
        
        # Skip rows without odds data
        if pd.isna(row.get('avg_odds')):
            continue
        
        # Determine the higher Elo fighter
        fighter_has_higher_elo = row['precomp_elo'] > row['opp_precomp_elo']
        
        # We always bet on the higher Elo fighter
        # If FIGHTER has higher Elo, use this row's odds directly
        # If opponent has higher Elo, we need to find the opponent's row for odds
        if fighter_has_higher_elo:
            # FIGHTER is our bet - use this row
            bet_on = row['FIGHTER']
            bet_against = row['opp_FIGHTER']
            bet_odds = row['avg_odds']
            # result=1 means FIGHTER won (our bet won)
            bet_won = (row['result'] == 1)
            elo_diff = row['precomp_elo'] - row['opp_precomp_elo']
        else:
            # Opponent has higher Elo, need to find opponent's row for their odds
            opp_key = (row['opp_FIGHTER'], row['FIGHTER'], str(row['DATE']))
            if opp_key not in odds_by_fight:
                continue
            bet_on = row['opp_FIGHTER']
            bet_against = row['FIGHTER']
            bet_odds = odds_by_fight[opp_key]
            # result=0 means FIGHTER lost, so opponent (our bet) won
            bet_won = (row['result'] == 0)
            elo_diff = row['opp_precomp_elo'] - row['precomp_elo']
        
        # Calculate expected probability from Elo (for log loss and brier score)
        # expected_prob is the probability we assign to our bet winning
        expected_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        
        # Convert American odds to decimal odds
        decimal_odds = american_odds_to_decimal(bet_odds)
        if decimal_odds is None:
            continue
        
        # Simulate $1 bet on the higher Elo fighter
        bet_amount = 1.0
        total_wagered += bet_amount
        
        if bet_won:
            payout = bet_amount * decimal_odds
        else:
            payout = 0
        
        total_returned += payout
        
        records.append({
            'date': row['DATE'],
            'bet_on': bet_on,
            'bet_against': bet_against,
            'bet_won': int(bet_won),
            'elo_diff': elo_diff,
            'expected_prob': expected_prob,
            'avg_odds': bet_odds,
            'decimal_odds': decimal_odds,
            'bet_amount': bet_amount,
            'payout': payout,
            'profit': payout - bet_amount
        })
    
    if not records:
        return {
            'roi_percent': None,
            'log_loss': None,
            'brier_score': None,
            'accuracy': None,
            'total_bets': 0,
            'records': pd.DataFrame()
        }
    
    records_df = pd.DataFrame(records)
    
    # Calculate ROI
    total_profit = total_returned - total_wagered
    roi_percent = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    
    # Calculate accuracy
    accuracy = records_df['bet_won'].mean()
    
    # Calculate Log Loss
    # Log loss = -1/N * sum(y * log(p) + (1-y) * log(1-p))
    # where y is actual result (bet_won) and p is predicted probability
    epsilon = 1e-10  # Small value to avoid log(0), using 1e-10 for numerical stability
    expected_probs = records_df['expected_prob'].clip(epsilon, 1 - epsilon)
    actual_results = records_df['bet_won']
    log_loss = -np.mean(
        actual_results * np.log(expected_probs) + 
        (1 - actual_results) * np.log(1 - expected_probs)
    )
    
    # Calculate Brier Score
    # Brier score = 1/N * sum((p - y)^2)
    # where y is actual result (bet_won) and p is predicted probability
    brier_score = np.mean((expected_probs - actual_results) ** 2)
    
    return {
        'roi_percent': roi_percent,
        'log_loss': log_loss,
        'brier_score': brier_score,
        'accuracy': accuracy,
        'total_bets': len(records),
        'total_wagered': total_wagered,
        'total_returned': total_returned,
        'total_profit': total_profit,
        'records': records_df
    }


def display_roi_metrics(roi_results):
    """
    Display ROI metrics in a formatted way.
    
    Args:
        roi_results: Dictionary returned by compute_roi_predictions()
    """
    print("\n" + "="*60)
    print("ROI & BETTING METRICS (Elo-based predictions)")
    print("="*60)
    
    if roi_results['total_bets'] == 0:
        print("No valid bets to analyze.")
        return
    
    print(f"\nTotal Bets: {roi_results['total_bets']}")
    print(f"Total Wagered: ${roi_results['total_wagered']:.2f}")
    print(f"Total Returned: ${roi_results['total_returned']:.2f}")
    print(f"Total Profit/Loss: ${roi_results['total_profit']:.2f}")
    print(f"\nROI: {roi_results['roi_percent']:.2f}%")
    print(f"Accuracy: {roi_results['accuracy']:.4f} ({roi_results['accuracy']*100:.2f}%)")
    print(f"\nLog Loss: {roi_results['log_loss']:.4f}")
    print(f"Brier Score: {roi_results['brier_score']:.4f}")
    print("="*60)


def compute_roi_over_time(roi_results, group_by='event'):
    """
    Calculate ROI over time, broken down by event or time period.
    
    Args:
        roi_results: Dictionary returned by compute_roi_predictions() containing 'records' DataFrame
        group_by: How to group results - 'event' (by date/event), 'month', or 'year'
    
    Returns:
        DataFrame with cumulative ROI metrics over time
    """
    records_df = roi_results.get('records')
    if records_df is None or records_df.empty:
        return pd.DataFrame()
    
    # Sort by date
    records_df = records_df.copy()
    records_df['date'] = pd.to_datetime(records_df['date'])
    records_df = records_df.sort_values('date')
    
    # Create grouping column based on group_by parameter
    if group_by == 'month':
        records_df['period'] = records_df['date'].dt.to_period('M').astype(str)
    elif group_by == 'year':
        records_df['period'] = records_df['date'].dt.year.astype(str)
    else:  # 'event' - group by date (each event)
        records_df['period'] = records_df['date'].dt.date.astype(str)
    
    # Calculate metrics by period
    period_stats = records_df.groupby('period').agg({
        'bet_amount': 'sum',
        'payout': 'sum',
        'profit': 'sum',
        'bet_won': ['sum', 'count', 'mean'],
        'expected_prob': 'mean'
    }).reset_index()
    
    # Flatten column names
    period_stats.columns = ['period', 'wagered', 'returned', 'profit', 
                           'wins', 'bets', 'accuracy', 'avg_expected_prob']
    
    # Calculate period ROI
    period_stats['roi_percent'] = (period_stats['profit'] / period_stats['wagered']) * 100
    
    # Calculate cumulative metrics
    period_stats['cumulative_wagered'] = period_stats['wagered'].cumsum()
    period_stats['cumulative_returned'] = period_stats['returned'].cumsum()
    period_stats['cumulative_profit'] = period_stats['profit'].cumsum()
    period_stats['cumulative_roi'] = (period_stats['cumulative_profit'] / period_stats['cumulative_wagered']) * 100
    period_stats['cumulative_bets'] = period_stats['bets'].cumsum()
    period_stats['cumulative_wins'] = period_stats['wins'].cumsum()
    period_stats['cumulative_accuracy'] = period_stats['cumulative_wins'] / period_stats['cumulative_bets']
    
    return period_stats


def display_roi_over_time(roi_over_time_df, show_all=False):
    """
    Display ROI over time in a formatted table.
    
    Args:
        roi_over_time_df: DataFrame returned by compute_roi_over_time()
        show_all: If True, show all periods. If False, show summary (first 5, last 5)
    """
    if roi_over_time_df.empty:
        print("No ROI data to display.")
        return
    
    print("\n" + "="*100)
    print("ROI OVER TIME - Cumulative Performance")
    print("="*100)
    
    # Format for display
    display_cols = ['period', 'bets', 'wins', 'accuracy', 'profit', 'roi_percent', 
                    'cumulative_bets', 'cumulative_profit', 'cumulative_roi']
    
    df_display = roi_over_time_df[display_cols].copy()
    df_display['accuracy'] = df_display['accuracy'].apply(lambda x: f"{x*100:.1f}%")
    df_display['roi_percent'] = df_display['roi_percent'].apply(lambda x: f"{x:.2f}%")
    df_display['cumulative_roi'] = df_display['cumulative_roi'].apply(lambda x: f"{x:.2f}%")
    df_display['profit'] = df_display['profit'].apply(lambda x: f"${x:.2f}")
    df_display['cumulative_profit'] = df_display['cumulative_profit'].apply(lambda x: f"${x:.2f}")
    
    df_display.columns = ['Period', 'Bets', 'Wins', 'Accuracy', 'Profit', 'ROI%', 
                          'Total Bets', 'Total Profit', 'Cumulative ROI%']
    
    if show_all or len(df_display) <= 15:
        print(df_display.to_string(index=False))
    else:
        # Show first 5 and last 5
        print("First 5 events:")
        print(df_display.head(5).to_string(index=False))
        print(f"\n... ({len(df_display) - 10} events hidden) ...\n")
        print("Last 5 events:")
        print(df_display.tail(5).to_string(index=False))
    
    print("="*100)
    
    # Summary statistics
    final_row = roi_over_time_df.iloc[-1]
    print(f"\nFinal Summary:")
    print(f"  Total Events: {len(roi_over_time_df)}")
    print(f"  Total Bets: {int(final_row['cumulative_bets'])}")
    print(f"  Total Profit: ${final_row['cumulative_profit']:.2f}")
    print(f"  Final Cumulative ROI: {final_row['cumulative_roi']:.2f}%")
    print(f"  Final Cumulative Accuracy: {final_row['cumulative_accuracy']*100:.2f}%")
    
    # Calculate streak info (use local calculation to avoid modifying input DataFrame)
    profitable_events = (roi_over_time_df['profit'] > 0).sum()
    losing_events = len(roi_over_time_df) - profitable_events
    print(f"\n  Profitable Events: {profitable_events} ({profitable_events/len(roi_over_time_df)*100:.1f}%)")
    print(f"  Losing Events: {losing_events} ({losing_events/len(roi_over_time_df)*100:.1f}%)")


def plot_roi_over_time(roi_over_time_df, save_path=None):
    """
    Plot ROI over time.
    
    Args:
        roi_over_time_df: DataFrame returned by compute_roi_over_time()
        save_path: Optional path to save the plot
    """
    if roi_over_time_df.empty:
        print("No ROI data to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cumulative ROI over time
    ax1 = axes[0, 0]
    ax1.plot(range(len(roi_over_time_df)), roi_over_time_df['cumulative_roi'], 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Event Number')
    ax1.set_ylabel('Cumulative ROI (%)')
    ax1.set_title('Cumulative ROI Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Profit over time
    ax2 = axes[0, 1]
    ax2.fill_between(range(len(roi_over_time_df)), roi_over_time_df['cumulative_profit'], 
                     alpha=0.3, color='blue')
    ax2.plot(range(len(roi_over_time_df)), roi_over_time_df['cumulative_profit'], 'b-', linewidth=2)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Event Number')
    ax2.set_ylabel('Cumulative Profit ($)')
    ax2.set_title('Cumulative Profit Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Per-event ROI (bar chart)
    ax3 = axes[1, 0]
    colors = ['green' if r >= 0 else 'red' for r in roi_over_time_df['roi_percent']]
    ax3.bar(range(len(roi_over_time_df)), roi_over_time_df['roi_percent'], color=colors, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_xlabel('Event Number')
    ax3.set_ylabel('ROI (%)')
    ax3.set_title('Per-Event ROI')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative Accuracy over time
    ax4 = axes[1, 1]
    ax4.plot(range(len(roi_over_time_df)), roi_over_time_df['cumulative_accuracy'] * 100, 'g-', linewidth=2)
    ax4.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
    ax4.set_xlabel('Event Number')
    ax4.set_ylabel('Cumulative Accuracy (%)')
    ax4.set_title('Cumulative Accuracy Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def compare_odds_sources(odds_df):
    """
    Compare different odds sources (avg_odds, draftkings_odds, fanduel_odds, betmgm_odds)
    in terms of accuracy, log loss, and brier score.
    
    This function evaluates how well each sportsbook's odds predict fight outcomes
    by calculating implied probabilities from the odds.
    
    Note: The prediction logic uses a 0.5 implied probability threshold, which doesn't
    account for the vig (sportsbook margin) built into odds. This is a simplification
    that assumes the favorite (implied probability > 50%) should win.
    
    Args:
        odds_df: DataFrame with fight data including various odds columns and result column
    
    Returns:
        dict: Dictionary containing metrics for each odds source
    """
    odds_sources = ['avg_odds', 'draftkings_odds', 'fanduel_odds', 'betmgm_odds']
    results = {}
    
    # Normalize dates
    df = odds_df.copy()
    df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    
    for odds_col in odds_sources:
        if odds_col not in df.columns:
            print(f"Warning: {odds_col} not found in DataFrame")
            continue
        
        records = []
        processed_fights = set()  # Track unique fights per odds source
        
        for _, row in df.iterrows():
            # Skip invalid results
            if row['result'] not in (0, 1):
                continue
            if pd.isna(row['DATE']):
                continue
            if pd.isna(row.get(odds_col)):
                continue
            
            # Create unique fight key
            fight_key = tuple(sorted([row['FIGHTER'], row['opp_FIGHTER']])) + (str(row['DATE']),)
            
            if fight_key in processed_fights:
                continue
            processed_fights.add(fight_key)
            
            # Get odds for this fighter
            odds = row[odds_col]
            decimal_odds = american_odds_to_decimal(odds)
            if decimal_odds is None or decimal_odds <= 1:
                continue
            
            # Calculate implied probability from odds
            # Implied probability = 1 / decimal_odds
            implied_prob = 1 / decimal_odds
            
            # Actual result: 1 = this fighter won, 0 = this fighter lost
            actual_result = int(row['result'])
            
            # Prediction: if implied_prob > 0.5, odds favor this fighter winning
            predicted = 1 if implied_prob > 0.5 else 0
            
            records.append({
                'fighter': row['FIGHTER'],
                'opponent': row['opp_FIGHTER'],
                'date': row['DATE'],
                'odds': odds,
                'implied_prob': implied_prob,
                'predicted': predicted,
                'result': actual_result,
                'correct': int(predicted == actual_result)
            })
        
        if not records:
            results[odds_col] = {
                'total_fights': 0,
                'accuracy': None,
                'log_loss': None,
                'brier_score': None
            }
            continue
        
        records_df = pd.DataFrame(records)
        
        # Calculate accuracy
        accuracy = records_df['correct'].mean()
        
        # Calculate Log Loss
        epsilon = 1e-10
        implied_probs = records_df['implied_prob'].clip(epsilon, 1 - epsilon)
        actual_results = records_df['result']
        log_loss = -np.mean(
            actual_results * np.log(implied_probs) + 
            (1 - actual_results) * np.log(1 - implied_probs)
        )
        
        # Calculate Brier Score
        brier_score = np.mean((implied_probs - actual_results) ** 2)
        
        results[odds_col] = {
            'total_fights': len(records),
            'accuracy': accuracy,
            'log_loss': log_loss,
            'brier_score': brier_score,
            'records': records_df
        }
    
    return results


def display_odds_comparison(comparison_results):
    """
    Display comparison of different odds sources in a formatted table.
    
    Args:
        comparison_results: Dictionary returned by compare_odds_sources()
    """
    print("\n" + "="*90)
    print("ODDS SOURCE COMPARISON - Accuracy, Log Loss, and Brier Score")
    print("="*90)
    print(f"\n{'Odds Source':<20} {'Fights':<10} {'Accuracy':<25} {'Log Loss':<15} {'Brier Score':<15}")
    print("-"*90)
    
    for source, metrics in comparison_results.items():
        if metrics['accuracy'] is None:
            print(f"{source:<20} {'N/A':<10} {'N/A':<25} {'N/A':<15} {'N/A':<15}")
        else:
            accuracy_str = f"{metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)"
            print(f"{source:<20} {metrics['total_fights']:<10} {accuracy_str:<25} {metrics['log_loss']:<15.4f} {metrics['brier_score']:<15.4f}")
    
    print("="*90)
    
    # Find best performers
    valid_results = {k: v for k, v in comparison_results.items() if v['accuracy'] is not None}
    if valid_results:
        best_accuracy = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        best_log_loss = min(valid_results.items(), key=lambda x: x[1]['log_loss'])
        best_brier = min(valid_results.items(), key=lambda x: x[1]['brier_score'])
        
        print(f"\nBest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']*100:.2f}%)")
        print(f"Best Log Loss: {best_log_loss[0]} ({best_log_loss[1]['log_loss']:.4f})")
        print(f"Best Brier Score: {best_brier[0]} ({best_brier[1]['brier_score']:.4f})")


def display_top_n_elos(df, n=10):
    #display the top n postcomp_elo values byt fighter
    # dont' display multiple instances of the same fighter
    print("Highest elo's ever achieved by a fighter:")
    history = build_fighter_history(df)
    sorted_cdf = history.sort_values('post_elo', ascending=False)
    displayed_fighters = []
    for _, index in sorted_cdf.iterrows():
        if index['fighter'] not in displayed_fighters:
            print(f"- {index['fighter']}: {index['post_elo']}")
            displayed_fighters.append(index['fighter'])
            if len(displayed_fighters) >= n:
                break
    return displayed_fighters

def most_recent_elo(df, n=100):
    #display top 10 postcomp_elo values by fighter based on their most recent fight
    print("Most recent elo's by a fighter:")
    history = build_fighter_history(df)
    history_sorted = history.sort_values('date')
    most_recent_df = history_sorted.groupby('fighter', as_index=False).last()
    sorted_df = most_recent_df.sort_values('post_elo', ascending=False).head(n)
    for _, row in sorted_df.iterrows():
        print(f"- {row['fighter']}: {row['post_elo']}")
    return sorted_df

def most_recent_elo_by_fighter(df, fighter, n=100):
    #display the most recent elo for a given fighter
    print(f"Most recent elo for {fighter}:")
    history = build_fighter_history(df)
    most_recent_df = history[history['fighter'] == fighter].sort_values('date', ascending=False).head(n)
    for _, row in most_recent_df.iterrows():
        print(f"- {row['date']}: {row['post_elo']}")
    return most_recent_df

def elo_accuracy(df, year=None):
    #calculate the accuracy of the elo predictions
    #elo_accuracy is the percentage of fights where the elo prediction was correct
    #if 'precomp_elo' is greater than 'opp_precomp_elo', and 'result' is 1, then the prediction was correct
    #if 'precomp_elo' is less than 'opp_precomp_elo', and 'result' is 0, then the prediction was correct
    #if 'precomp_elo' is greater than 'opp_precomp_elo', and 'result' is 0, then the prediction was incorrect
    #if 'precomp_elo' is less than 'opp_precomp_elo', and 'result' is 1, then the prediction was incorrect
    predictions = compute_fight_predictions(df)
    if predictions.empty:
        return None
    if year is not None:
        predictions = predictions[predictions['date'].dt.year >= year]
    if predictions.empty:
        return None
    return predictions['correct'].mean()

def plot_elo_accuracy_by_year(df):
    #plot the accuracy of the elo predictions by year
    #elo_accuracy_by_year is the percentage of fights where the elo prediction was correct by year
    #if 'precomp_elo' is greater than 'opp_precomp_elo', and 'result' is 1, then the prediction was correct
    #if 'precomp_elo' is less than 'opp_precomp_elo', and 'result' is 0, then the prediction was correct
    #if 'precomp_elo' is greater than 'opp_precomp_elo', and 'result' is 0, then the prediction was incorrect
    #if 'precomp_elo' is less than 'opp_precomp_elo', and 'result' is 1, then the prediction was incorrect
    predictions = compute_fight_predictions(df)
    if predictions.empty:
        print("No fights with valid results to calculate accuracy.")
        return
    accuracy_by_year = predictions.groupby(predictions['date'].dt.year)['correct'].mean()
    x_years = accuracy_by_year.index.to_numpy()
    y_accuracy = accuracy_by_year.to_numpy()
    plt.figure()
    plt.plot(x_years, y_accuracy, marker='o', label='Observed accuracy')
    if len(x_years) >= 2:
        slope, intercept = np.polyfit(x_years, y_accuracy, 1)
        future_year = 2026
        extended_years = np.append(x_years, future_year)
        line_x = np.linspace(extended_years.min(), extended_years.max(), 200)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, linestyle='--', color='orange', label='Trend line')
        future_accuracy = slope * future_year + intercept
        plt.scatter([future_year], [future_accuracy], color='red', label=f'2026 prediction: {future_accuracy:.2%}')
        print(f"Projected Elo accuracy for {future_year}: {future_accuracy:.2%}")
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.title('Elo Accuracy by Year')
    plt.legend()
    plt.show()

def plot_accuracy_by_event(df, ema_span=50):
    predictions = compute_fight_predictions(df)
    if predictions.empty:
        print("No fights with valid results to chart event-level accuracy.")
        return None
    predictions = predictions.sort_values('date').reset_index(drop=True)
    predictions['ema_accuracy'] = predictions['correct'].ewm(span=ema_span, adjust=False).mean()
    plt.figure()
    plt.scatter(
        predictions['date'],
        predictions['correct'],
        alpha=0.3,
        label='Fight outcome (1 = predicted correctly)',
    )
    plt.plot(
        predictions['date'],
        predictions['ema_accuracy'],
        color='orange',
        label=f'EMA accuracy (span={ema_span})',
    )
    plt.xlabel('Event date')
    plt.ylabel('Prediction accuracy')
    plt.title('Elo Prediction Accuracy by Event')
    plt.legend()
    plt.show()
    next_accuracy = predictions['ema_accuracy'].iloc[-1]
    print(f"Predicted accuracy for the next event (EMA span {ema_span}): {next_accuracy:.2%}")
    return next_accuracy

def plot_monthly_accuracy_with_ema(df, ema_span=6):
    predictions = compute_fight_predictions(df)
    if predictions.empty:
        print("No fights with valid results to chart monthly accuracy.")
        return None
    predictions = predictions.sort_values('date').reset_index(drop=True)
    predictions['month'] = predictions['date'].dt.to_period('M').dt.to_timestamp()
    monthly_accuracy = predictions.groupby('month')['correct'].mean().sort_index()
    if monthly_accuracy.empty:
        print("No monthly accuracy data available.")
        return None
    ema_series = monthly_accuracy.ewm(span=ema_span, adjust=False).mean()
    ema_prediction_next = ema_series.shift(1)
    monthly_df = pd.DataFrame(
        {
            'actual_accuracy': monthly_accuracy,
            'ema_smooth': ema_series,
            'ema_prediction_next': ema_prediction_next,
        }
    )
    monthly_df['absolute_error'] = (
        monthly_df['ema_prediction_next'] - monthly_df['actual_accuracy']
    ).abs()
    valid_errors = monthly_df.dropna(subset=['ema_prediction_next'])
    if not valid_errors.empty:
        mae = valid_errors['absolute_error'].mean()
        print(f"EMA (span={ema_span}) next-month MAE: {mae:.2%}")
    next_month_prediction = ema_series.iloc[-1] if not ema_series.empty else None
    plt.figure()
    plt.plot(
        monthly_df.index,
        monthly_df['actual_accuracy'],
        marker='o',
        label='Actual monthly accuracy',
    )
    plt.plot(
        monthly_df.index,
        monthly_df['ema_prediction_next'],
        marker='x',
        linestyle='--',
        label='EMA next-month prediction',
    )
    plt.xlabel('Month')
    plt.ylabel('Accuracy')
    plt.title(f'Elo Accuracy vs EMA Prediction (Monthly, span={ema_span})')
    plt.legend()
    plt.show()
    if next_month_prediction is not None:
        print(
            f"Predicted accuracy for the next month (EMA span {ema_span}): "
            f"{next_month_prediction:.2%}"
        )
    return next_month_prediction

def graph_fighter_elo_history(df, fighter):
    history = build_fighter_history(df)
    history = history[history['fighter'] == fighter]
    history = history.sort_values('date')
    
    if history.empty:
        print(f"No history found for fighter: {fighter}")
        return history
    
    plt.figure(figsize=(14, 7))
    
    # Plot line connecting post-comp Elo across fights (overall trend)
    plt.plot(history['date'], history['post_elo'], 'b-', alpha=0.3, linewidth=1.5, label='Post-fight Elo trend')
    
    # For each fight, draw a line from pre-comp to post-comp to show the change
    for idx, row in history.iterrows():
        plt.plot([row['date'], row['date']], [row['pre_elo'], row['post_elo']], 
                'g-', alpha=0.6, linewidth=2, zorder=3)
    
    # Plot pre-comp Elo points (before each fight)
    plt.scatter(history['date'], history['pre_elo'], c='orange', s=60, alpha=0.8, 
               zorder=5, marker='o', edgecolors='darkorange', linewidths=1.5, label='Pre-fight Elo')
    
    # Plot post-comp Elo points (after each fight)
    plt.scatter(history['date'], history['post_elo'], c='red', s=60, alpha=0.8, 
               zorder=5, marker='s', edgecolors='darkred', linewidths=1.5, label='Post-fight Elo')
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Elo Rating', fontsize=12)
    plt.title(f'Elo History for {fighter} ({len(history)} fights)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print(f"Total fights plotted for {fighter}: {len(history)}")
    return history


def get_method_of_victory(row):
    """
    Determine the method of victory from fight data.
    
    The data contains paired flags for each outcome type:
    - 'ko'/'kod': KO/TKO win/loss respectively
    - 'subw'/'subwd': Submission win/loss
    - 'udec'/'udecd': Unanimous decision win/loss
    - 'sdec'/'sdecd': Split decision win/loss  
    - 'mdec'/'mdecd': Majority decision win/loss
    
    We check both flags to determine the method regardless of which fighter won.
    
    Args:
        row: DataFrame row containing MOV flags
    
    Returns:
        String describing the method of victory
    """
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
    return 'Unknown'


def _normalize_datetime(dt_series):
    """
    Normalize a datetime series by removing timezone information if present.
    
    Handles both timezone-aware and timezone-naive datetimes safely.
    
    Args:
        dt_series: pandas Series of datetime values
    
    Returns:
        pandas Series with timezone-naive datetimes
    """
    dt_series = pd.to_datetime(dt_series)
    # Check if the series has timezone info
    if dt_series.dt.tz is not None:
        return dt_series.dt.tz_localize(None)
    return dt_series


def analyze_random_events(df, roi_results, odds_df=None, n_events=5, random_seed=None):
    """
    Randomly select and analyze n unique events, showing ALL valid fights from each event.
    
    This function provides in-depth fight-by-fight analysis for randomly selected events,
    including both fights that were bet on (from ROI records) and other valid fights that
    occurred on the same date.
    
    A valid fight is one where:
    - Both fighters have at least one prior fight (history)
    - The fight result is valid (0 or 1, no draws)
    - Both fighters have Elo ratings
    
    Args:
        df: DataFrame with fight data including precomp_elo, postcomp_elo, etc.
        roi_results: Dictionary returned by compute_roi_predictions() containing 'records' DataFrame
        odds_df: Optional DataFrame with odds data for additional fight details
        n_events: Number of random events to analyze (default: 5)
        random_seed: Optional random seed for reproducibility
    
    Returns:
        List of dictionaries, each containing detailed analysis for one event:
        - event_date: Date of the event
        - event_name: Name of the event (if available)
        - bets: List of bet details for fights that were bet on
        - other_fights: List of other valid fights that weren't bet on
        - event_roi: ROI for this specific event (from bets only)
        - win_count: Number of winning bets
        - loss_count: Number of losing bets
        - total_profit: Net profit for the event
    """
    records_df = roi_results.get('records')
    if records_df is None or records_df.empty:
        return []
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Get unique event dates
    records_df = records_df.copy()
    records_df['date'] = pd.to_datetime(records_df['date'])
    unique_dates = records_df['date'].dt.date.unique()
    
    # Randomly select n_events
    n_events = min(n_events, len(unique_dates))
    selected_dates = np.random.choice(unique_dates, size=n_events, replace=False)
    
    # Build lookup for Elo data from df
    df_copy = df.copy()
    df_copy['DATE'] = _normalize_datetime(df_copy['DATE'])
    
    # Build fighter history for prior fight check
    history = build_fighter_history(df_copy)
    first_fight_dates = history.groupby('fighter')['date'].min().to_dict()
    
    # Create lookup dictionary for Elo and fight details
    # Build bidirectionally so lookup works regardless of fighter order
    elo_lookup = {}
    for _, row in df_copy.iterrows():
        key = (row['FIGHTER'], row['opp_FIGHTER'], str(row['DATE'].date()))
        elo_data = {
            'precomp_elo': row.get('precomp_elo'),
            'postcomp_elo': row.get('postcomp_elo'),
            'opp_precomp_elo': row.get('opp_precomp_elo'),
            'opp_postcomp_elo': row.get('opp_postcomp_elo'),
            'event_name': row.get('EVENT', 'Unknown Event'),
            'method_of_victory': get_method_of_victory(row),
            'round': row.get('round'),
            'time_format': row.get('time_format'),
            'result': row.get('result'),
            'weight_of_fight': row.get('weight_of_fight')
        }
        elo_lookup[key] = elo_data
        # Add reverse key so lookup works regardless of fighter order
        reverse_key = (row['opp_FIGHTER'], row['FIGHTER'], str(row['DATE'].date()))
        if reverse_key not in elo_lookup:
            elo_lookup[reverse_key] = elo_data
    
    # Build lookup for odds if odds_df is provided
    # Build bidirectionally so lookup works regardless of fighter order
    odds_lookup = {}
    if odds_df is not None:
        odds_df_copy = odds_df.copy()
        odds_df_copy['DATE'] = _normalize_datetime(odds_df_copy['DATE'])
        for _, row in odds_df_copy.iterrows():
            key = (row['FIGHTER'], row['opp_FIGHTER'], str(row['DATE'].date()))
            if key not in odds_lookup:
                odds_data = {
                    'event_name': row.get('EVENT', 'Unknown Event'),
                    'avg_odds': row.get('avg_odds'),
                    'draftkings_odds': row.get('draftkings_odds'),
                    'fanduel_odds': row.get('fanduel_odds'),
                    'betmgm_odds': row.get('betmgm_odds')
                }
                odds_lookup[key] = odds_data
                # Add reverse key so lookup works regardless of fighter order
                reverse_key = (row['opp_FIGHTER'], row['FIGHTER'], str(row['DATE'].date()))
                if reverse_key not in odds_lookup:
                    odds_lookup[reverse_key] = odds_data
    
    # Build set of bet fight keys for quick lookup
    bet_fight_keys = set()
    for _, bet in records_df.iterrows():
        bet_date_str = str(pd.to_datetime(bet['date']).date())
        # Add both orderings to catch the fight regardless of row perspective
        bet_fight_keys.add((bet['bet_on'], bet['bet_against'], bet_date_str))
        bet_fight_keys.add((bet['bet_against'], bet['bet_on'], bet_date_str))
    
    event_analyses = []
    
    for event_date in sorted(selected_dates):
        event_date_str = str(event_date)
        event_bets = records_df[records_df['date'].dt.date == event_date]
        
        # Get all fights from df on this date
        event_fights_df = df_copy[df_copy['DATE'].dt.date == event_date]
        
        # Determine event name from the df or odds_lookup
        event_name = 'Unknown Event'
        for _, row in event_fights_df.iterrows():
            if row.get('EVENT') and str(row.get('EVENT')) != 'nan':
                event_name = row['EVENT']
                break
        
        bets_list = []
        for _, bet in event_bets.iterrows():
            bet_on = bet['bet_on']
            bet_against = bet['bet_against']
            
            # Try to get Elo data - first try bet_on as fighter
            key1 = (bet_on, bet_against, event_date_str)
            key2 = (bet_against, bet_on, event_date_str)
            
            elo_data = elo_lookup.get(key1, elo_lookup.get(key2, {}))
            odds_data = odds_lookup.get(key1, odds_lookup.get(key2, {}))
            
            # Determine pre/post Elo for both fighters
            if key1 in elo_lookup:
                bet_on_pre_elo = elo_data.get('precomp_elo')
                bet_on_post_elo = elo_data.get('postcomp_elo')
                bet_against_pre_elo = elo_data.get('opp_precomp_elo')
                bet_against_post_elo = elo_data.get('opp_postcomp_elo')
            else:
                bet_on_pre_elo = elo_data.get('opp_precomp_elo')
                bet_on_post_elo = elo_data.get('opp_postcomp_elo')
                bet_against_pre_elo = elo_data.get('precomp_elo')
                bet_against_post_elo = elo_data.get('postcomp_elo')
            
            # Calculate Elo changes
            bet_on_elo_change = None
            bet_against_elo_change = None
            if bet_on_pre_elo is not None and bet_on_post_elo is not None:
                bet_on_elo_change = bet_on_post_elo - bet_on_pre_elo
            if bet_against_pre_elo is not None and bet_against_post_elo is not None:
                bet_against_elo_change = bet_against_post_elo - bet_against_pre_elo
            
            bet_details = {
                'bet_on': bet_on,
                'bet_against': bet_against,
                'bet_on_pre_elo': bet_on_pre_elo,
                'bet_on_post_elo': bet_on_post_elo,
                'bet_on_elo_change': bet_on_elo_change,
                'bet_against_pre_elo': bet_against_pre_elo,
                'bet_against_post_elo': bet_against_post_elo,
                'bet_against_elo_change': bet_against_elo_change,
                'elo_diff': bet['elo_diff'],
                'avg_odds_american': bet['avg_odds'],
                'decimal_odds': bet['decimal_odds'],
                'bet_amount': bet['bet_amount'],
                'payout': bet['payout'],
                'profit': bet['profit'],
                'bet_won': bool(bet['bet_won']),
                'expected_prob': bet['expected_prob'],
                'method_of_victory': elo_data.get('method_of_victory', 'Unknown'),
                'round': elo_data.get('round'),
                'draftkings_odds': odds_data.get('draftkings_odds'),
                'fanduel_odds': odds_data.get('fanduel_odds'),
                'betmgm_odds': odds_data.get('betmgm_odds'),
                'bet_placed': True
            }
            bets_list.append(bet_details)
        
        # Now find other valid fights on this date that weren't bet on
        other_fights_list = []
        processed_fight_keys = set()  # Track processed fights to avoid duplicates
        
        for _, row in event_fights_df.iterrows():
            fighter = row['FIGHTER']
            opponent = row['opp_FIGHTER']
            fight_key = tuple(sorted([fighter, opponent]))
            
            # Skip if already processed (avoid duplicate from both perspectives)
            if fight_key in processed_fight_keys:
                continue
            processed_fight_keys.add(fight_key)
            
            # Skip if this fight was already bet on
            key1 = (fighter, opponent, event_date_str)
            key2 = (opponent, fighter, event_date_str)
            if key1 in bet_fight_keys or key2 in bet_fight_keys:
                continue
            
            # Apply filtering criteria:
            # 1. Valid result (0 or 1, no draws)
            if row['result'] not in (0, 1):
                continue
            
            # 2. Both fighters have prior fight history
            if not has_prior_history(first_fight_dates, fighter, row['DATE']):
                continue
            if not has_prior_history(first_fight_dates, opponent, row['DATE']):
                continue
            
            # 3. Both fighters have Elo ratings
            if pd.isna(row.get('precomp_elo')) or pd.isna(row.get('opp_precomp_elo')):
                continue
            
            # Get fight details
            elo_data = elo_lookup.get(key1, {})
            odds_data = odds_lookup.get(key1, odds_lookup.get(key2, {}))
            
            # Determine winner based on result and Elo information
            fighter_won = (row['result'] == 1)
            winner = fighter if fighter_won else opponent
            loser = opponent if fighter_won else fighter
            
            # Calculate Elo change for both fighters
            fighter_pre_elo = row.get('precomp_elo')
            fighter_post_elo = row.get('postcomp_elo')
            opp_pre_elo = row.get('opp_precomp_elo')
            opp_post_elo = row.get('opp_postcomp_elo')
            
            fighter_elo_change = None
            opp_elo_change = None
            if fighter_pre_elo is not None and fighter_post_elo is not None:
                fighter_elo_change = fighter_post_elo - fighter_pre_elo
            if opp_pre_elo is not None and opp_post_elo is not None:
                opp_elo_change = opp_post_elo - opp_pre_elo
            
            # Calculate Elo diff (higher Elo - lower Elo)
            elo_diff = None
            if fighter_pre_elo is not None and opp_pre_elo is not None:
                elo_diff = abs(fighter_pre_elo - opp_pre_elo)
            
            # Determine reason why bet wasn't placed
            # At this point, the fight passed all filtering criteria (valid result, prior history, Elo)
            # The reason it wasn't bet on is either:
            # 1. No odds available
            # 2. Equal Elo ratings (can't determine which fighter to bet on)
            # 3. Other filtering in compute_roi_predictions
            if not odds_data.get('avg_odds'):
                reason_no_bet = 'No odds available'
            elif fighter_pre_elo == opp_pre_elo:
                reason_no_bet = 'Equal Elo ratings'
            else:
                reason_no_bet = 'Odds data mismatch'
            
            fight_details = {
                'fighter': fighter,
                'opponent': opponent,
                'winner': winner,
                'loser': loser,
                'fighter_won': fighter_won,
                'fighter_pre_elo': fighter_pre_elo,
                'fighter_post_elo': fighter_post_elo,
                'fighter_elo_change': fighter_elo_change,
                'opp_pre_elo': opp_pre_elo,
                'opp_post_elo': opp_post_elo,
                'opp_elo_change': opp_elo_change,
                'elo_diff': elo_diff,
                'avg_odds_american': odds_data.get('avg_odds'),
                'method_of_victory': elo_data.get('method_of_victory', get_method_of_victory(row)),
                'round': row.get('round'),
                'draftkings_odds': odds_data.get('draftkings_odds'),
                'fanduel_odds': odds_data.get('fanduel_odds'),
                'betmgm_odds': odds_data.get('betmgm_odds'),
                'bet_placed': False,
                'reason_no_bet': reason_no_bet
            }
            other_fights_list.append(fight_details)
        
        # Calculate event-level metrics (from bets only)
        total_wagered = sum(b['bet_amount'] for b in bets_list) if bets_list else 0
        total_payout = sum(b['payout'] for b in bets_list) if bets_list else 0
        total_profit = total_payout - total_wagered
        event_roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
        win_count = sum(1 for b in bets_list if b['bet_won'])
        loss_count = len(bets_list) - win_count
        
        event_analyses.append({
            'event_date': event_date,
            'event_name': event_name,
            'bets': bets_list,
            'other_fights': other_fights_list,
            'event_roi': event_roi,
            'win_count': win_count,
            'loss_count': loss_count,
            'total_wagered': total_wagered,
            'total_payout': total_payout,
            'total_profit': total_profit
        })
    
    return event_analyses


def display_detailed_event_analysis(event_analyses):
    """
    Display detailed analysis for each event in a well-formatted manner.
    
    Formats and prints:
    - Well-structured event headers with date and event name
    - Tabular display of all bets with Elo matchups, odds, and outcomes
    - Other valid fights that weren't bet on
    - Fighter Elo progression visualization
    - Event profitability summary
    - Individual bet profitability breakdown
    
    Args:
        event_analyses: List of event analysis dictionaries returned by analyze_random_events()
    """
    if not event_analyses:
        print("\nNo event data to display.")
        return
    
    print("\n" + "="*100)
    print("DETAILED EVENT ANALYSIS - Random Sample of {} Events".format(len(event_analyses)))
    print("="*100)
    
    for i, event in enumerate(event_analyses, 1):
        # Event Header
        print("\n" + "-"*100)
        print(f"EVENT {i}/{len(event_analyses)}")
        print("-"*100)
        print(f"Date: {event['event_date']}")
        print(f"Event: {event['event_name']}")
        
        # Get other_fights count safely (backwards compatible)
        other_fights = event.get('other_fights', [])
        total_valid_fights = len(event['bets']) + len(other_fights)
        
        print(f"Total Valid Fights: {total_valid_fights} ({len(event['bets'])} bets placed, {len(other_fights)} not bet on)")
        print(f"Betting Record: {event['win_count']}W - {event['loss_count']}L")
        
        # Event Summary (betting results)
        if event['total_wagered'] > 0:
            profit_str = f"${event['total_profit']:+.2f}"
            roi_str = f"{event['event_roi']:+.2f}%"
            status = "PROFIT" if event['total_profit'] > 0 else ("LOSS" if event['total_profit'] < 0 else "BREAK EVEN")
            print(f"Betting Result: {status} | Wagered: ${event['total_wagered']:.2f} | "
                  f"Returned: ${event['total_payout']:.2f} | Profit: {profit_str} | ROI: {roi_str}")
        
        # Section 1: Bets Placed
        if event['bets']:
            print("\n  === BETS PLACED ({} fights) ===".format(len(event['bets'])))
            print("\n  {:<25} {:<25} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
                "Bet On", "Bet Against", "Pre-Elo", "Opp Elo", "Odds", "Bet", "Payout", "Result"
            ))
            print("  " + "-"*96)
            
            # Individual Bet Details
            for bet in event['bets']:
                # Format Elo values
                pre_elo_str = f"{bet['bet_on_pre_elo']:.0f}" if bet['bet_on_pre_elo'] is not None else "N/A"
                opp_elo_str = f"{bet['bet_against_pre_elo']:.0f}" if bet['bet_against_pre_elo'] is not None else "N/A"
                
                # Format odds (American) - convert to float if it's a string
                if pd.notna(bet['avg_odds_american']):
                    try:
                        odds_value = float(bet['avg_odds_american'])
                        odds_str = f"{odds_value:+.0f}"
                    except (ValueError, TypeError):
                        odds_str = str(bet['avg_odds_american'])
                else:
                    odds_str = "N/A"
                
                # Result indicator
                result_str = "WIN" if bet['bet_won'] else "LOSS"
                payout_str = f"${bet['payout']:.2f}"
                
                # Truncate fighter names if too long
                bet_on_name = bet['bet_on'][:23] + ".." if len(bet['bet_on']) > 25 else bet['bet_on']
                bet_against_name = bet['bet_against'][:23] + ".." if len(bet['bet_against']) > 25 else bet['bet_against']
                
                print("  {:<25} {:<25} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10}".format(
                    bet_on_name,
                    bet_against_name,
                    pre_elo_str,
                    opp_elo_str,
                    odds_str,
                    f"${bet['bet_amount']:.2f}",
                    payout_str,
                    result_str
                ))
        
        # Section 2: Other Valid Fights (not bet on)
        if other_fights:
            print("\n  === OTHER VALID FIGHTS ({} fights - not bet on) ===".format(len(other_fights)))
            print("\n  {:<25} {:<25} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
                "Fighter", "Opponent", "Pre-Elo", "Opp Elo", "Odds", "Winner", "Why No Bet"
            ))
            print("  " + "-"*96)
            
            for fight in other_fights:
                # Format Elo values
                fighter_elo_str = f"{fight['fighter_pre_elo']:.0f}" if fight['fighter_pre_elo'] is not None else "N/A"
                opp_elo_str = f"{fight['opp_pre_elo']:.0f}" if fight['opp_pre_elo'] is not None else "N/A"
                
                # Format odds (American)
                odds_str = f"{fight['avg_odds_american']:+.0f}" if pd.notna(fight.get('avg_odds_american')) else "N/A"
                
                # Winner indicator
                winner_str = fight['winner'][:10] if len(fight['winner']) > 10 else fight['winner']
                
                # Truncate fighter names if too long
                fighter_name = fight['fighter'][:23] + ".." if len(fight['fighter']) > 25 else fight['fighter']
                opp_name = fight['opponent'][:23] + ".." if len(fight['opponent']) > 25 else fight['opponent']
                
                # Reason for no bet
                reason = fight.get('reason_no_bet', 'Unknown')[:12]
                
                print("  {:<25} {:<25} {:>10} {:>10} {:>12} {:>10} {:>12}".format(
                    fighter_name,
                    opp_name,
                    fighter_elo_str,
                    opp_elo_str,
                    odds_str,
                    winner_str,
                    reason
                ))
        
        # Elo Change Summary
        print("\n  Elo Changes (Bets):")
        for bet in event['bets']:
            if bet.get('bet_on_elo_change') is not None:
                change_str = f"{bet['bet_on_elo_change']:+.1f}"
                win_indicator = "(Won)" if bet['bet_won'] else "(Lost)"
                print(f"    {bet['bet_on']}: {change_str} {win_indicator}")
        
        if other_fights:
            print("\n  Elo Changes (Other Fights):")
            for fight in other_fights:
                if fight.get('fighter_elo_change') is not None:
                    change_str = f"{fight['fighter_elo_change']:+.1f}"
                    win_indicator = "(Won)" if fight['fighter_won'] else "(Lost)"
                    print(f"    {fight['fighter']}: {change_str} {win_indicator}")
        
        # Additional fight details
        print("\n  Fight Details:")
        for bet in event['bets']:
            mov = bet.get('method_of_victory', 'Unknown')
            round_num = bet.get('round')
            winner = bet['bet_on'] if bet['bet_won'] else bet['bet_against']
            
            round_str = f"Round {round_num}" if round_num else ""
            print(f"    [BET] {bet['bet_on']} vs {bet['bet_against']}: Winner - {winner} ({mov}) {round_str}")
        
        for fight in other_fights:
            mov = fight.get('method_of_victory', 'Unknown')
            round_num = fight.get('round')
            
            round_str = f"Round {round_num}" if round_num else ""
            print(f"    [---] {fight['fighter']} vs {fight['opponent']}: Winner - {fight['winner']} ({mov}) {round_str}")
    
    # Overall Summary
    print("\n" + "="*100)
    print("OVERALL SUMMARY")
    print("="*100)
    
    total_bets = sum(len(e['bets']) for e in event_analyses)
    total_other_fights = sum(len(e.get('other_fights', [])) for e in event_analyses)
    total_wins = sum(e['win_count'] for e in event_analyses)
    total_losses = sum(e['loss_count'] for e in event_analyses)
    total_wagered = sum(e['total_wagered'] for e in event_analyses)
    total_profit = sum(e['total_profit'] for e in event_analyses)
    overall_roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    
    profitable_events = sum(1 for e in event_analyses if e['total_profit'] > 0)
    
    print(f"Events Analyzed: {len(event_analyses)}")
    print(f"Total Valid Fights: {total_bets + total_other_fights} ({total_bets} bets placed, {total_other_fights} not bet on)")
    print(f"Profitable Events: {profitable_events} ({profitable_events/len(event_analyses)*100:.1f}%)")
    print(f"Betting Record: {total_wins}W - {total_losses}L ({total_wins/total_bets*100:.1f}% win rate)" if total_bets > 0 else "Betting Record: No bets")
    print(f"Total Wagered: ${total_wagered:.2f}")
    print(f"Total Profit/Loss: ${total_profit:+.2f}")
    print(f"Overall ROI: {overall_roi:+.2f}%")
    print("="*100)

    
if __name__ == "__main__":
    df = pd.read_csv('data/interleaved_cleaned.csv')
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    df = run_basic_elo(df)
    print(display_top_n_elos(df, n = 30))
    print(most_recent_elo(df, n = 30))
    print(elo_accuracy(df))
    #run basic elo with mov
    df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    # Use custom weights from best GA run
    df = run_basic_elo_with_mov(df, 
                                k=237.43990534337982,
                                w_ko=1.055465364523305,
                                w_sub=1.7461659394019622,
                                w_udec=1.0084238164539632,
                                w_sdec=0.6050612748940885,
                                w_mdec=0.8479369152159737)
    print(display_top_n_elos(df, n = 100))
    print(most_recent_elo(df, n = 100))
    print(elo_accuracy(df))
    '''
    #plot_elo_accuracy_by_year(df)
    #next_event_accuracy = plot_accuracy_by_event(df)
    if next_event_accuracy is not None:
        print(f"Next event accuracy prediction (EMA): {next_event_accuracy:.2%}")
    next_month_accuracy = plot_monthly_accuracy_with_ema(df)
    if next_month_accuracy is not None:
        print(f"Next month accuracy prediction (EMA): {next_month_accuracy:.2%}")
    '''

    #plot the elo history for a given fighter
    #graph_fighter_elo_history(df, fighter = 'Hamdy Abdelwahab')

    most_recent_elo_df = most_recent_elo_by_fighter(df, fighter = 'Alexa Grasso')
    most_recent_elo_df.to_csv('data/most_recent_elo.csv', index=False)

    graph_fighter_elo_history(df, fighter = 'Chris Barnett')
    most_recent_elo_df = most_recent_elo_by_fighter(df, fighter = 'Natalia Silva')
    most_recent_elo_df.to_csv('data/most_recent_elo.csv', index=False)
    
    # ROI Calculator - uses Elo from main.py calculations, odds from after_averaging.csv
    print("\n" + "="*60)
    print("RUNNING ROI CALCULATOR")
    print("Using Elo ratings calculated above, odds from after_averaging.csv")
    print("="*60)
    
    # Load the after_averaging.csv file which has the odds data only
    odds_df = pd.read_csv('after_averaging.csv', low_memory=False)
    odds_df['DATE'] = pd.to_datetime(odds_df['DATE'])
    
    # Use 'df' which already has Elo values calculated by run_basic_elo_with_mov above
    # Merge odds from after_averaging.csv into df
    roi_results = compute_roi_predictions(df, odds_df=odds_df)
    display_roi_metrics(roi_results)
    
    # ROI Over Time Analysis - track performance by event
    print("\n" + "="*60)
    print("ROI OVER TIME ANALYSIS")
    print("="*60)
    roi_over_time = compute_roi_over_time(roi_results, group_by='event')
    display_roi_over_time(roi_over_time, show_all=False)
    
    # Plot ROI over time
    plot_roi_over_time(roi_over_time, save_path='images/roi_over_time.png')
    
    # Compare different odds sources (avg_odds, draftkings_odds, fanduel_odds, betmgm_odds)
    print("\n" + "="*60)
    print("COMPARING ODDS SOURCES")
    print("="*60)
    comparison_results = compare_odds_sources(odds_df)
    display_odds_comparison(comparison_results)
    
    # Detailed Event Analysis - analyze 5 randomly selected events
    print("\n" + "="*60)
    print("DETAILED EVENT ANALYSIS")
    print("Randomly selecting 5 events for in-depth fight-by-fight analysis")
    print("="*60)
    event_analyses = analyze_random_events(df, roi_results, odds_df=odds_df, n_events=5)
    display_detailed_event_analysis(event_analyses)