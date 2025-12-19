import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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


def plot_ga_optimization_history(history_df, optimize_for="accuracy", save_path=None, show_plot=True):
    """
    Plot the optimization history from a genetic algorithm run.
    
    Creates two subplots:
    1. Fitness over generations (best, average, worst)
    2. Parameter evolution over generations (k-factor and denominator)
    
    Args:
        history_df: DataFrame with columns 'generation', 'best_fitness', 'avg_fitness', 
                   'worst_fitness', and 'best_genes' (dict with 'k' and 'denominator')
        optimize_for: Optimization mode ("accuracy", "roi", "log_loss", "composite")
        save_path: Optional path to save the figure (e.g., "ga_optimization.png")
        show_plot: Whether to display the plot (default: True)
        
    Returns:
        Figure object
    """
    if history_df.empty:
        print("Warning: No history data to plot")
        return None
    
    # Extract parameters from best_genes dictionary
    k_values = []
    denominator_values = []
    for _, row in history_df.iterrows():
        genes = row['best_genes']
        k_values.append(genes.get('k', np.nan))
        denominator_values.append(genes.get('denominator', np.nan))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Fitness over generations
    generations = history_df['generation']
    
    # Determine y-axis label based on optimization mode
    if optimize_for == "roi":
        ylabel = "ROI (as decimal)"
        title_suffix = "ROI"
    elif optimize_for == "log_loss":
        ylabel = "Fitness (exp(-log_loss))"
        title_suffix = "Log Loss"
    elif optimize_for == "composite":
        ylabel = "Composite Fitness"
        title_suffix = "Composite Fitness"
    else:  # accuracy
        ylabel = "Accuracy"
        title_suffix = "Accuracy"
    
    ax1.plot(generations, history_df['best_fitness'], 'g-', linewidth=2.5, 
             label='Best', marker='o', markersize=6, markerfacecolor='lightgreen', 
             markeredgecolor='darkgreen', markeredgewidth=1.5, alpha=0.9)
    ax1.plot(generations, history_df['avg_fitness'], 'b--', linewidth=2, 
             label='Average', marker='s', markersize=5, markerfacecolor='lightblue',
             markeredgecolor='darkblue', markeredgewidth=1.5, alpha=0.8)
    ax1.plot(generations, history_df['worst_fitness'], 'r:', linewidth=2, 
             label='Worst', marker='^', markersize=5, markerfacecolor='lightcoral',
             markeredgecolor='darkred', markeredgewidth=1.5, alpha=0.7)
    
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax1.set_title(f'Genetic Algorithm Optimization Progress - {title_suffix}', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Add value annotations for first and last generation
    if len(generations) > 0:
        # First generation
        ax1.annotate(f'{history_df["best_fitness"].iloc[0]:.4f}',
                    xy=(generations.iloc[0], history_df['best_fitness'].iloc[0]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
        # Last generation
        ax1.annotate(f'{history_df["best_fitness"].iloc[-1]:.4f}',
                    xy=(generations.iloc[-1], history_df['best_fitness'].iloc[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7, fontweight='bold')
    
    # Subplot 2: Parameter evolution
    ax2_twin = ax2.twinx()  # Create twin axis for denominator
    
    # Plot K-factor on left y-axis
    line1 = ax2.plot(generations, k_values, 'o-', color='#1f77b4', linewidth=2.5,
                     label='K-factor', marker='o', markersize=6, 
                     markerfacecolor='lightblue', markeredgecolor='darkblue',
                     markeredgewidth=1.5, alpha=0.9)
    ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax2.set_ylabel('K-factor', fontsize=12, fontweight='bold', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    
    # Plot denominator on right y-axis
    line2 = ax2_twin.plot(generations, denominator_values, 's-', color='#ff7f0e', 
                          linewidth=2.5, label='Denominator', marker='s', markersize=6,
                          markerfacecolor='#ffd580', markeredgecolor='#ff7f0e',
                          markeredgewidth=1.5, alpha=0.9)
    ax2_twin.set_ylabel('Denominator', fontsize=12, fontweight='bold', color='#ff7f0e')
    ax2_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    
    ax2.set_title('Parameter Evolution Over Generations', fontsize=14, 
                  fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)
    
    # Add final parameter values as text
    if len(generations) > 0:
        final_k = k_values[-1]
        final_denom = denominator_values[-1]
        ax2.text(0.02, 0.98, f'Final K: {final_k:.2f}\nFinal Denom: {final_denom:.2f}',
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        except (IOError, OSError) as e:
            print(f"\nWarning: Could not save plot to {save_path}: {e}")
    
    # Show if requested
    if show_plot:
        plt.show()
    
    return fig
