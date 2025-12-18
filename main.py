import pandas as pd
import numpy as np
import argparse
from elo.calculator import run_basic_elo, run_basic_elo_with_mov
from elo.visualization import display_top_n_elos, most_recent_elo, graph_fighter_elo_history
from elo.elo_utils import add_bout_counts

# Minimum probability for betting odds calculation (prevents extreme payouts)
MIN_BET_PROBABILITY = 0.01

def calculate_roi(df, denominator=400, confidence_threshold=50, validation_percentile=0.8, use_bout_filter=False):
    """
    Calculate ROI for the Elo predictions using the same method as GA optimization.
    
    Args:
        df: DataFrame with fight data including precomp_elo columns
        denominator: Elo denominator for calculating win probabilities
        confidence_threshold: Minimum confidence threshold for betting (default 50)
        validation_percentile: Use fights after this percentile for validation (default 0.8)
        use_bout_filter: If True, filter out fights where either fighter has no prior history (default False)
    
    Returns:
        Dictionary with ROI metrics
    """
    # Split data for validation
    cutoff = df["DATE"].quantile(validation_percentile)
    val_df = df[df["DATE"] > cutoff].copy()
    
    # Extract predictions and actuals
    predictions = []
    actuals = []
    
    for _, row in val_df.iterrows():
        # Filter valid fights
        if row.get("result") not in (0, 1):
            continue
        if pd.isna(row.get("DATE")):
            continue
        if row.get("precomp_elo") == row.get("opp_precomp_elo"):
            continue
        
        # Check bout counts (same as GA optimization)
        if use_bout_filter:
            bout1 = row.get("precomp_boutcount", 0)
            bout2 = row.get("opp_precomp_boutcount", 0)
            if pd.isna(bout1) or pd.isna(bout2) or bout1 < 1 or bout2 < 1:
                continue
        
        # Calculate prediction probability
        elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / denominator))
        
        predictions.append(pred_prob)
        actuals.append(int(row["result"]))
    
    if len(predictions) == 0:
        # Return worst possible ROI when no valid predictions
        # -100% means complete loss of all wagered money
        return {
            'roi': -1.0,
            'roi_percent': -100.0,
            'total_bets': 0,
            'total_profit': 0.0,
            'accuracy': 0.0
        }
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate ROI
    total_profit = 0.0
    total_bets = 0
    
    for pred, actual in zip(predictions, actuals):
        # Use prediction confidence to determine if we should bet
        confidence = abs(pred - 0.5) * 2  # Scale to 0-1 range
        # Scale by 1000 to match typical Elo difference magnitude
        if confidence * 1000 >= confidence_threshold:
            # Determine predicted winner and their win probability
            pred_winner = 1 if pred > 0.5 else 0
            pred_prob = pred if pred > 0.5 else (1 - pred)
            
            # Calculate realistic payout based on implied odds
            pred_prob_clamped = max(pred_prob, MIN_BET_PROBABILITY)
            payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
            
            # We always bet 1 unit
            if pred_winner == actual:
                # Win: get back bet + payout
                total_profit += payout_multiplier
            else:
                # Lose: lose the bet
                total_profit -= 1.0
            total_bets += 1
    
    roi = (total_profit / total_bets) if total_bets > 0 else -1.0
    
    # Calculate accuracy
    pred_labels = (predictions > 0.5).astype(int)
    accuracy = np.mean(pred_labels == actuals)
    
    return {
        'roi': roi,
        'roi_percent': roi * 100,
        'total_bets': total_bets,
        'total_profit': total_profit,
        'total_wagered': float(total_bets),
        'accuracy': accuracy,
        'total_predictions': len(predictions)
    }

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run Elo rating system with customizable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default parameters
  python main.py
  
  # Custom K-factor and denominator
  python main.py --k 10.0 --denominator 437.78
  
  # Show ROI calculation (matches GA optimization setup)
  python main.py --k 10.0 --denominator 437.78 --show-roi --use-ga-setup
  
  # Evaluate on recent fights but use full Elo history (NEW DEFAULT)
  python main.py --k 10.0 --denominator 437.78 --show-roi --use-ga-setup --tail-fights 3000
  
  # Match exact GA optimization behavior (calculate Elo only on tail)
  python main.py --k 10.0 --denominator 437.78 --show-roi --use-ga-setup --tail-fights 3000 --tail-only
  
  # Adjust confidence threshold for betting
  python main.py --k 10.0 --denominator 437.78 --show-roi --confidence-threshold 30
        """
    )
    parser.add_argument('--k', type=float, default=32.0,
                        help='K-factor for basic Elo (default: 32.0)')
    parser.add_argument('--denominator', type=float, default=400.0,
                        help='Denominator for Elo calculations (default: 400.0)')
    parser.add_argument('--show-roi', action='store_true',
                        help='Calculate and display ROI metrics')
    parser.add_argument('--confidence-threshold', type=float, default=50.0,
                        help='Confidence threshold for betting in ROI calculation (default: 50.0)')
    parser.add_argument('--validation-percentile', type=float, default=0.8,
                        help='Percentile for validation split in ROI calculation (default: 0.8)')
    parser.add_argument('--use-ga-setup', action='store_true',
                        help='Use GA optimization setup (adds bout counts, filters by prior history)')
    parser.add_argument('--tail-fights', type=int, default=None,
                        help='Display/evaluate only the last N fights (Elo still calculated on full history unless --tail-only)')
    parser.add_argument('--tail-only', action='store_true',
                        help='When used with --tail-fights, calculate Elo ONLY on tail fights (matches old GA behavior)')
    
    args = parser.parse_args()
    
    # Load full dataset
    df_full = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
    df_full['result'] = pd.to_numeric(df_full['result'], errors='coerce')
    df_full['DATE'] = pd.to_datetime(df_full['DATE'])
    df_full = df_full.sort_values('DATE').reset_index(drop=True)
    
    # Handle tail_fights option
    if args.tail_fights:
        if args.tail_only:
            # Old behavior: Use ONLY tail fights for Elo calculation (matches GA optimization)
            print(f"\nUsing ONLY last {args.tail_fights} fights for Elo calculation (GA style)")
            df_full = df_full.tail(args.tail_fights).copy()
        else:
            # New behavior: Calculate Elo on full dataset, but display/evaluate only tail
            print(f"\nCalculating Elo on full dataset ({len(df_full)} fights)")
            print(f"Will display/evaluate only last {args.tail_fights} fights")
            # Mark which fights are in the tail for later filtering
            tail_start_idx = len(df_full) - args.tail_fights
            df_full['in_tail'] = False
            df_full.loc[tail_start_idx:, 'in_tail'] = True
    
    # Add bout counts if using GA setup
    if args.use_ga_setup:
        df_full = add_bout_counts(df_full)
        if "precomp_boutcount" in df_full.columns:
            df_full["precomp_boutcount"] = pd.to_numeric(df_full["precomp_boutcount"], errors="coerce")
        if "opp_precomp_boutcount" in df_full.columns:
            df_full["opp_precomp_boutcount"] = pd.to_numeric(df_full["opp_precomp_boutcount"], errors="coerce")
    
    print("\n" + "="*60)
    print("ELO RATINGS - Basic System")
    print("="*60)
    print(f"Parameters: K={args.k:.4f}, Denominator={args.denominator:.4f}")
    print("="*60)
    
    # Run Elo calculation
    df_basic_full = run_basic_elo(df_full.copy(), k=args.k, denominator=args.denominator)
    
    # If using tail without --tail-only, filter to display only tail fights
    if args.tail_fights and not args.tail_only:
        df_basic = df_basic_full[df_basic_full['in_tail']].copy()
        print(f"\nShowing results for last {args.tail_fights} fights")
        print(f"(Elo ratings calculated from full {len(df_basic_full)} fight history)")
    else:
        df_basic = df_basic_full
    
    display_top_n_elos(df_basic, n=10)
    most_recent_elo(df_basic, n=10)
    
    # Calculate and display ROI if requested
    if args.show_roi:
        print("\n" + "="*60)
        print("ROI CALCULATION")
        print("="*60)
        print(f"Confidence Threshold: {args.confidence_threshold:.2f}")
        print(f"Validation Split: Last {(1-args.validation_percentile)*100:.0f}% of data")
        if args.use_ga_setup:
            print("Using GA setup: Filtering by bout counts (both fighters must have prior history)")
        print("="*60)
        
        roi_metrics = calculate_roi(
            df_basic,
            denominator=args.denominator,
            confidence_threshold=args.confidence_threshold,
            validation_percentile=args.validation_percentile,
            use_bout_filter=args.use_ga_setup
        )
        
        print(f"\nValidation Set Predictions: {roi_metrics['total_predictions']}")
        print(f"Total Bets Placed: {roi_metrics['total_bets']}")
        print(f"Total Wagered: ${roi_metrics['total_wagered']:.2f}")
        print(f"Total Profit: ${roi_metrics['total_profit']:.2f}")
        print(f"ROI: {roi_metrics['roi_percent']:.2f}%")
        print(f"Prediction Accuracy: {roi_metrics['accuracy']*100:.2f}%")
        
        print("\nNote: ROI is displayed as a percentage of return on investment.")
        print("      For example, 34.92% means $1 bet returns $1.3492 on average.")
        print("      Negative ROI means losses exceed winnings.")
    
    print("\n" + "="*60)
    print("ELO RATINGS - With Method of Victory")
    print("="*60)
    # Note: MOV Elo uses pre-optimized fixed parameters from full_genetic_with_k_denom_mov.py
    # These are separate from the basic Elo parameters and are kept fixed for consistency
    df_mov_full = run_basic_elo_with_mov(
        df_full.copy(),
        k=135.5024295855922,
        w_ko=1.304905948911245,
        w_sub=2.0,
        w_udec=0.8150291233970635,
        w_sdec=0.6277860597403592,
        w_mdec=1.0229206801627735
    )
    
    # Filter to tail if specified (and not using tail-only mode)
    if args.tail_fights and not args.tail_only:
        df_mov = df_mov_full[df_mov_full['in_tail']].copy()
    else:
        df_mov = df_mov_full
    
    display_top_n_elos(df_mov, n=10)
    most_recent_elo(df_mov, n=10)
    
    fighter = input("\nEnter fighter name to view Elo history (or press Enter to skip): ").strip()
    if fighter:
        graph_fighter_elo_history(df_mov, fighter)

if __name__ == "__main__":
    main()
