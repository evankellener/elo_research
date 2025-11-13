import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def run_basic_elo(df, k = 32, base_elo = 1500, denominator = 400):
    ratings = {}
    pre_elo, post_elos = [], []
    opp_pre_elos, opp_post_elos = [], []

    for _, row in df.iterrows():
        f1, f2, result = row['FIGHTER'], row['opp_FIGHTER'], row['result']

        ratings.setdefault(f1, base_elo)
        ratings.setdefault(f2, base_elo)

        f1_pre, f2_pre = ratings[f1], ratings[f2]

        expected_f1 = 1/(1+10.0**((f2_pre-f1_pre)/denominator))
        expected_f2 = 1/(1+10.0**((f1_pre-f2_pre)/denominator))

        new_f1_rating = f1_pre + k * (result - expected_f1)
        new_f2_rating = f2_pre + k * (1 - result - expected_f2)

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

def build_fighter_history(df):
    fighter_history = pd.concat(
        [
            df[['DATE', 'FIGHTER', 'precomp_elo', 'postcomp_elo']].rename(
                columns={'FIGHTER': 'fighter', 'precomp_elo': 'pre_elo', 'postcomp_elo': 'post_elo'}
            ),
            df[['DATE', 'opp_FIGHTER', 'opp_precomp_elo', 'opp_postcomp_elo']].rename(
                columns={'opp_FIGHTER': 'fighter', 'opp_precomp_elo': 'pre_elo', 'opp_postcomp_elo': 'post_elo'}
            ),
        ],
        ignore_index=True,
    )
    fighter_history = fighter_history.sort_values('DATE').rename(columns={'DATE': 'date'}).reset_index(drop=True)
    return fighter_history

def has_prior_history(first_fight_dates, fighter, date):
    first_date = first_fight_dates.get(fighter)
    if first_date is None:
        return False
    return date > first_date

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

def most_recent_elo(df, n=10):
    #display top 10 postcomp_elo values by fighter based on their most recent fight
    print("Most recent elo's by a fighter:")
    history = build_fighter_history(df)
    history_sorted = history.sort_values('date')
    most_recent_df = history_sorted.groupby('fighter', as_index=False).last()
    sorted_df = most_recent_df.sort_values('post_elo', ascending=False).head(n)
    for _, row in sorted_df.iterrows():
        print(f"- {row['fighter']}: {row['post_elo']}")
    return sorted_df

def most_recent_elo_by_fighter(df, fighter, n=10):
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
    
if __name__ == "__main__":
    df = pd.read_csv('interleaved_cleaned.csv')
    df['result'] = pd.to_numeric(df['result'], errors='coerce')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)
    df = run_basic_elo(df)
    print(display_top_n_elos(df, n = 30))
    print(most_recent_elo(df, n = 30))
    print(elo_accuracy(df))
    plot_elo_accuracy_by_year(df)
    next_event_accuracy = plot_accuracy_by_event(df)
    if next_event_accuracy is not None:
        print(f"Next event accuracy prediction (EMA): {next_event_accuracy:.2%}")
    next_month_accuracy = plot_monthly_accuracy_with_ema(df)
    if next_month_accuracy is not None:
        print(f"Next month accuracy prediction (EMA): {next_month_accuracy:.2%}")


