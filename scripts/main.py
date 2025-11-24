import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from elo_utils import method_of_victory_scale, build_fighter_history, has_prior_history

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
    graph_fighter_elo_history(df, fighter = 'Hamdy Abdelwahab')

    most_recent_elo_df = most_recent_elo_by_fighter(df, fighter = 'Hamdy Abdelwahab')
    most_recent_elo_df.to_csv('data/most_recent_elo.csv', index=False)

    graph_fighter_elo_history(df, fighter = 'Chris Barnett')
    most_recent_elo_df = most_recent_elo_by_fighter(df, fighter = 'Chris Barnett')
    most_recent_elo_df.to_csv('data/most_recent_elo.csv', index=False)