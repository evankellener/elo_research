import matplotlib.pyplot as plt
import pandas as pd
from .elo_utils import build_fighter_history

def display_top_n_elos(df, n=10):
    print(f"\nTop {n} highest Elo ratings ever achieved:")
    history = build_fighter_history(df)
    sorted_cdf = history.sort_values('post_elo', ascending=False)
    displayed_fighters = []
    for _, index in sorted_cdf.iterrows():
        if index['fighter'] not in displayed_fighters:
            print(f"  {index['fighter']}: {index['post_elo']:.0f}")
            displayed_fighters.append(index['fighter'])
            if len(displayed_fighters) >= n:
                break
    return displayed_fighters

def most_recent_elo(df, n=100):
    print(f"\nTop {n} most recent Elo ratings:")
    history = build_fighter_history(df)
    history_sorted = history.sort_values('date')
    most_recent_df = history_sorted.groupby('fighter', as_index=False).last()
    sorted_df = most_recent_df.sort_values('post_elo', ascending=False).head(n)
    for _, row in sorted_df.iterrows():
        print(f"  {row['fighter']}: {row['post_elo']:.0f}")
    return sorted_df

def graph_fighter_elo_history(df, fighter):
    history = build_fighter_history(df)
    history = history[history['fighter'] == fighter]
    history = history.sort_values('date')
    
    if history.empty:
        print(f"No history found for fighter: {fighter}")
        return history
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(history['date'], history['post_elo'], 'b-', alpha=0.3, linewidth=1.5, label='Post-fight Elo trend')
    
    for idx, row in history.iterrows():
        plt.plot([row['date'], row['date']], [row['pre_elo'], row['post_elo']], 
                'g-', alpha=0.6, linewidth=2, zorder=3)
    
    plt.scatter(history['date'], history['pre_elo'], c='orange', s=60, alpha=0.8, 
               zorder=5, marker='o', edgecolors='darkorange', linewidths=1.5, label='Pre-fight Elo')
    
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
