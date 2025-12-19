import pandas as pd
import numpy as np
import argparse
import random
from collections import defaultdict
from elo.calculator import run_basic_elo, run_basic_elo_with_mov
from elo.visualization import display_top_n_elos, most_recent_elo, graph_fighter_elo_history
from elo.elo_utils import add_bout_counts

# Minimum probability for betting odds calculation (prevents extreme payouts)
# 0.01 = 1% probability = 99:1 maximum odds
MIN_BET_PROBABILITY = 0.01

# Confidence scaling factor to convert probability confidence (0-1) to Elo points
# An Elo difference of 100 points translates to roughly 14% confidence
# This scaling (1000) allows confidence_threshold in Elo-equivalent units
# e.g., confidence_threshold=50 means bet when confidence > 5% (50/1000)
CONFIDENCE_SCALE_FACTOR = 1000

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
    
    # Calculate ROI by simulating betting strategy and tracking all predictions
    total_profit = 0.0
    total_bets = 0
    bet_events = []  # Track individual betting events
    all_fights = []  # Track ALL fights in validation set (for event display)
    predictions = []
    actuals = []
    
    for idx, row in val_df.iterrows():
        # Filter valid fights
        if row.get("result") not in (0, 1):
            continue
        if pd.isna(row.get("DATE")):
            continue
        if row.get("precomp_elo") == row.get("opp_precomp_elo"):
            continue
        
        # Check bout counts (same as GA optimization)
        # Filter to only include fights where both fighters have had more than one fight
        # (as specified in elo_explanatoin.md: "more than one fight")
        if use_bout_filter:
            bout1 = row.get("precomp_boutcount", 0)
            bout2 = row.get("opp_precomp_boutcount", 0)
            if pd.isna(bout1) or pd.isna(bout2) or bout1 <= 1 or bout2 <= 1:
                continue
        
        # Calculate prediction probability
        elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
        pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / denominator))
        
        # Track all predictions for accuracy calculation
        predictions.append(pred_prob)
        actuals.append(int(row["result"]))
        
        # Use prediction confidence to determine if we should bet
        confidence = abs(pred_prob - 0.5) * 2  # Scale to 0-1 range (0 = unsure, 1 = certain)
        
        # Store all fight information (whether bet or not)
        fight_info = {
            'event_name': row.get('EVENT', 'Unknown Event'),
            'event_date': row.get('DATE', None),
            'fighter': row.get('FIGHTER', 'Unknown'),
            'opponent': row.get('opp_FIGHTER', 'Unknown'),
            'fighter_elo': row.get('precomp_elo', 0),
            'opponent_elo': row.get('opp_precomp_elo', 0),
            'pred_prob': pred_prob,
            'confidence': confidence,
            'predicted_winner': 1 if pred_prob > 0.5 else 0,
            'actual_winner': int(row["result"]),
            'bet_placed': False,
            'won_bet': None,
            'payout_multiplier': None,
            'profit': None
        }
        
        # Check if confidence exceeds threshold (scaled to match Elo-equivalent units)
        if confidence * CONFIDENCE_SCALE_FACTOR >= confidence_threshold:
            # Determine predicted winner and their win probability
            predicted_winner = 1 if pred_prob > 0.5 else 0
            predicted_win_probability = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
            
            # Calculate realistic payout based on implied odds
            predicted_win_prob_clamped = max(predicted_win_probability, MIN_BET_PROBABILITY)
            payout_multiplier = (1.0 / predicted_win_prob_clamped) - 1.0
            
            # Determine outcome
            actual_winner = int(row["result"])
            won_bet = (predicted_winner == actual_winner)
            
            # Calculate profit for this bet
            if won_bet:
                profit = payout_multiplier
                total_profit += payout_multiplier
            else:
                profit = -1.0
                total_profit -= 1.0
            
            total_bets += 1
            
            # Update fight info with bet details
            fight_info.update({
                'bet_placed': True,
                'won_bet': won_bet,
                'payout_multiplier': payout_multiplier,
                'profit': profit
            })
            
            # Store bet event details (for backward compatibility)
            bet_events.append({
                'event_name': row.get('EVENT', 'Unknown Event'),
                'event_date': row.get('DATE', None),
                'fighter': row.get('FIGHTER', 'Unknown'),
                'opponent': row.get('opp_FIGHTER', 'Unknown'),
                'fighter_elo': row.get('precomp_elo', 0),
                'opponent_elo': row.get('opp_precomp_elo', 0),
                'pred_prob': pred_prob,
                'confidence': confidence,
                'predicted_winner': predicted_winner,
                'actual_winner': actual_winner,
                'won_bet': won_bet,
                'payout_multiplier': payout_multiplier,
                'profit': profit,
                'event_roi': profit  # Same as profit, kept for semantic clarity when displaying ROI
            })
        
        # Add to all fights list
        all_fights.append(fight_info)
    
    if len(predictions) == 0:
        # Return worst possible ROI when no valid predictions
        # -100% means complete loss of all wagered money
        return {
            'roi': -1.0,
            'roi_percent': -100.0,
            'total_bets': 0,
            'total_profit': 0.0,
            'accuracy': 0.0,
            'total_predictions': 0,
            'bet_events': [],
            'all_fights': []
        }
    
    roi = (total_profit / total_bets) if total_bets > 0 else -1.0
    
    # Calculate accuracy on all predictions (convert to numpy for vectorization)
    predictions_arr = np.array(predictions)
    actuals_arr = np.array(actuals)
    pred_labels = (predictions_arr > 0.5).astype(int)
    accuracy = np.mean(pred_labels == actuals_arr)
    
    return {
        'roi': roi,
        'roi_percent': roi * 100,
        'total_bets': total_bets,
        'total_profit': total_profit,
        'total_wagered': float(total_bets),
        'accuracy': accuracy,
        'total_predictions': len(predictions_arr),
        'bet_events': bet_events,
        'all_fights': all_fights
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
        
        # Display 5 random betting events to validate ROI
        if roi_metrics['all_fights']:
            print("\n" + "="*60)
            print("SAMPLE OF 5 RANDOM BETTING EVENTS")
            print("="*60)
            
            # Group all fights by UFC event name
            # Create unique fight identifiers to avoid showing same fight twice (once per fighter row)
            events_by_name = defaultdict(lambda: {})
            for fight in roi_metrics['all_fights']:
                event_name = fight['event_name']
                # Create a unique fight key using sorted fighter names
                fighters_sorted = tuple(sorted([fight['fighter'], fight['opponent']]))
                fight_key = fighters_sorted
                
                # Only add if we haven't seen this fight yet, or if this one has a bet placed
                if fight_key not in events_by_name[event_name] or fight['bet_placed']:
                    events_by_name[event_name][fight_key] = fight
            
            # Select up to 5 random UFC events
            event_names = list(events_by_name.keys())
            num_samples = min(5, len(event_names))
            sample_event_names = random.sample(event_names, num_samples)
            
            for i, event_name in enumerate(sample_event_names, 1):
                fights_dict = events_by_name[event_name]
                fights_in_event = list(fights_dict.values())
                
                # Calculate event-level statistics (only for fights with bets)
                event_total_bets = sum(1 for f in fights_in_event if f['bet_placed'])
                event_wins = sum(1 for f in fights_in_event if f['bet_placed'] and f['won_bet'])
                event_profit = sum(f['profit'] for f in fights_in_event if f['bet_placed'])
                event_roi = (event_profit / event_total_bets * 100) if event_total_bets > 0 else 0.0
                
                # Get event date from first fight
                event_date = fights_in_event[0].get('event_date', 'Unknown Date')
                
                print(f"\n{'='*60}")
                print(f"Event {i}: {event_name}")
                print(f"Date: {event_date}")
                print(f"Total Fights: {len(fights_in_event)} | Total Bets: {event_total_bets} | Wins: {event_wins} | Losses: {event_total_bets - event_wins}")
                if event_total_bets > 0:
                    print(f"Event Profit: ${event_profit:.2f} | Event ROI: {event_roi:.2f}%")
                print(f"{'='*60}")
                
                # Display each fight in the event
                for j, fight in enumerate(fights_in_event, 1):
                    print(f"\n  Fight {j}:")
                    print(f"    Match: {fight['fighter']} vs {fight['opponent']}")
                    print(f"    Elo Ratings: {fight['fighter_elo']:.1f} vs {fight['opponent_elo']:.1f}")
                    
                    # Show who was predicted to win
                    if fight['predicted_winner'] == 1:
                        predicted_fighter = fight['fighter']
                        pred_prob_display = fight['pred_prob'] * 100
                    else:
                        predicted_fighter = fight['opponent']
                        pred_prob_display = (1 - fight['pred_prob']) * 100
                    
                    print(f"    Predicted Winner: {predicted_fighter} ({pred_prob_display:.1f}% win probability)")
                    print(f"    Betting Confidence: {fight['confidence']*100:.1f}%")
                    
                    # Show bet information if bet was placed
                    if fight['bet_placed']:
                        print(f"    Payout Odds: {fight['payout_multiplier']:.2f}x (bet $1 to win ${fight['payout_multiplier']:.2f})")
                        
                        # Show actual outcome
                        if fight['actual_winner'] == 1:
                            actual_winner_name = fight['fighter']
                        else:
                            actual_winner_name = fight['opponent']
                        print(f"    Actual Winner: {actual_winner_name}")
                        
                        # Show bet result
                        if fight['won_bet']:
                            print(f"    Result: WON - Profit: ${fight['profit']:.2f}")
                        else:
                            print(f"    Result: LOST - Loss: ${abs(fight['profit']):.2f}")
                        print(f"    Fight ROI: {fight['profit']*100:.2f}%")
                    else:
                        # No bet placed - show why
                        # Show actual outcome
                        if fight['actual_winner'] == 1:
                            actual_winner_name = fight['fighter']
                        else:
                            actual_winner_name = fight['opponent']
                        print(f"    Actual Winner: {actual_winner_name}")
                        print(f"    Bet Status: NO BET (confidence below threshold)")
            
            print("\n" + "="*60)
        
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
