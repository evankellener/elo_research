"""
Test suite for boutcount filtering in ROI calculations.

This ensures that ROI and accuracy calculations only include fights where
both fighters have more than one fight (boutcount > 1), as specified in
elo_explanatoin.md.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from elo.calculator import run_basic_elo
from elo.elo_utils import add_bout_counts
from main import calculate_roi


def test_calculate_roi_filters_boutcount():
    """Test that calculate_roi properly filters fights by boutcount > 1."""
    # Create test data
    dates = pd.date_range('2020-01-01', periods=100, freq='W')
    test_data = []
    
    # Add fights with various boutcount scenarios
    for i in range(100):
        test_data.append({
            'DATE': dates[i],
            'FIGHTER': f'Fighter_{i % 20}',
            'opp_FIGHTER': f'Opponent_{i % 20}',
            'result': i % 2,
            'EVENT': f'Event_{i // 10}',
            'precomp_boutcount': (i % 5),  # Will have 0, 1, 2, 3, 4
            'opp_precomp_boutcount': ((i + 1) % 5),
            'precomp_elo': 1500 + (i % 100),
            'opp_precomp_elo': 1500 - (i % 100),
        })
    
    df = pd.DataFrame(test_data)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Run calculate_roi with bout filter
    roi_metrics = calculate_roi(
        df,
        denominator=400,
        confidence_threshold=50.0,
        validation_percentile=0.8,
        use_bout_filter=True
    )
    
    # Get validation split to manually verify
    cutoff = df['DATE'].quantile(0.8)
    val_df = df[df['DATE'] > cutoff].copy()
    
    # Count expected predictions (both fighters must have boutcount > 1)
    expected_predictions = val_df[
        (val_df['result'].isin([0, 1])) &
        (~pd.isna(val_df['DATE'])) &
        (val_df['precomp_elo'] != val_df['opp_precomp_elo']) &
        (val_df['precomp_boutcount'] > 1) &
        (val_df['opp_precomp_boutcount'] > 1)
    ]
    
    print(f"Test Results:")
    print(f"  Expected predictions: {len(expected_predictions)}")
    print(f"  Actual predictions: {roi_metrics['total_predictions']}")
    
    # Verify that predictions match expected count
    assert roi_metrics['total_predictions'] == len(expected_predictions), \
        f"Expected {len(expected_predictions)} predictions, got {roi_metrics['total_predictions']}"
    
    # Verify that all bet events have boutcount > 1
    for bet_event in roi_metrics['bet_events']:
        fighter = bet_event['fighter']
        opponent = bet_event['opponent']
        date = bet_event['event_date']
        
        fight = df[
            (df['FIGHTER'] == fighter) & 
            (df['opp_FIGHTER'] == opponent) &
            (df['DATE'] == date)
        ]
        
        if len(fight) > 0:
            fight_row = fight.iloc[0]
            assert fight_row['precomp_boutcount'] > 1, \
                f"Fighter {fighter} has boutcount <= 1"
            assert fight_row['opp_precomp_boutcount'] > 1, \
                f"Opponent {opponent} has boutcount <= 1"
    
    print("  ✓ All tests passed!")
    return True


def test_boutcount_filtering_with_real_data():
    """Test boutcount filtering with actual dataset (if available)."""
    try:
        # Load real data
        df = pd.read_csv('data/interleaved_cleaned.csv', low_memory=False)
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE'])
        df = df.sort_values('DATE').reset_index(drop=True)
        
        # Use a smaller subset for testing
        df_test = df.tail(1000).copy()
        df_test = add_bout_counts(df_test)
        df_test['precomp_boutcount'] = pd.to_numeric(df_test['precomp_boutcount'], errors='coerce')
        df_test['opp_precomp_boutcount'] = pd.to_numeric(df_test['opp_precomp_boutcount'], errors='coerce')
        
        # Run Elo
        df_with_elo = run_basic_elo(df_test, k=32, denominator=400)
        
        # Test with bout filter
        roi_metrics = calculate_roi(
            df_with_elo,
            denominator=400,
            confidence_threshold=50.0,
            validation_percentile=0.8,
            use_bout_filter=True
        )
        
        # Verify all predictions have boutcount > 1
        cutoff = df_with_elo['DATE'].quantile(0.8)
        val_df = df_with_elo[df_with_elo['DATE'] > cutoff].copy()
        
        # Count fights with boutcount <= 1 that should be excluded
        excluded_fights = val_df[
            (val_df['result'].isin([0, 1])) &
            ((val_df['precomp_boutcount'] <= 1) | (val_df['opp_precomp_boutcount'] <= 1))
        ]
        
        print(f"\nReal Data Test Results:")
        print(f"  Total validation fights: {len(val_df[val_df['result'].isin([0, 1])])}")
        print(f"  Excluded (boutcount <= 1): {len(excluded_fights)}")
        print(f"  Expected predictions: {len(val_df[val_df['result'].isin([0, 1])]) - len(excluded_fights)}")
        print(f"  Actual predictions: {roi_metrics['total_predictions']}")
        
        # The actual predictions should not include any fights with boutcount <= 1
        expected_without_excluded = len(val_df[val_df['result'].isin([0, 1])]) - len(excluded_fights)
        
        # Note: There may be additional filters (like equal Elos), so actual <= expected
        assert roi_metrics['total_predictions'] <= expected_without_excluded, \
            "More predictions than expected after boutcount filter"
        
        print("  ✓ Real data test passed!")
        return True
        
    except FileNotFoundError:
        print("\nReal Data Test: Skipped (data file not found)")
        return True


if __name__ == '__main__':
    print("="*60)
    print("BOUTCOUNT FILTERING TESTS")
    print("="*60)
    
    print("\n1. Testing with synthetic data...")
    test_calculate_roi_filters_boutcount()
    
    print("\n2. Testing with real data...")
    test_boutcount_filtering_with_real_data()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
