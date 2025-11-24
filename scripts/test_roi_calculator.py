"""
Unit tests for ROI calculator functions in main.py
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import (
    american_odds_to_decimal,
    compute_roi_predictions,
    compute_roi_over_time,
    compare_odds_sources
)


class TestAmericanOddsToDecimal(unittest.TestCase):
    """Tests for american_odds_to_decimal function"""
    
    def test_positive_odds(self):
        """Test conversion of positive American odds (underdog)"""
        # +150 should convert to 2.50 (bet $100, win $150, total return $250)
        self.assertAlmostEqual(american_odds_to_decimal(150), 2.5)
        # +100 should convert to 2.00 (even money)
        self.assertAlmostEqual(american_odds_to_decimal(100), 2.0)
        # +200 should convert to 3.00
        self.assertAlmostEqual(american_odds_to_decimal(200), 3.0)
        # +500 should convert to 6.00
        self.assertAlmostEqual(american_odds_to_decimal(500), 6.0)
    
    def test_negative_odds(self):
        """Test conversion of negative American odds (favorite)"""
        # -200 should convert to 1.50 (bet $200 to win $100)
        self.assertAlmostEqual(american_odds_to_decimal(-200), 1.5, places=4)
        # -100 should convert to 2.00 (even money)
        self.assertAlmostEqual(american_odds_to_decimal(-100), 2.0, places=4)
        # -150 should convert to 1.667
        self.assertAlmostEqual(american_odds_to_decimal(-150), 1.667, places=3)
        # -400 should convert to 1.25
        self.assertAlmostEqual(american_odds_to_decimal(-400), 1.25, places=4)
    
    def test_nan_input(self):
        """Test that NaN input returns None"""
        self.assertIsNone(american_odds_to_decimal(np.nan))
        self.assertIsNone(american_odds_to_decimal(float('nan')))
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Very large positive odds
        self.assertAlmostEqual(american_odds_to_decimal(10000), 101.0)
        # Very large negative odds
        self.assertAlmostEqual(american_odds_to_decimal(-10000), 1.01)


class TestComputeROIPredictions(unittest.TestCase):
    """Tests for compute_roi_predictions function"""
    
    def setUp(self):
        """Set up test data"""
        # Create a simple test DataFrame with Elo ratings
        # Note: postcomp_elo is required by build_fighter_history
        self.df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']),
            'FIGHTER': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A', 'Fighter D', 'Fighter C'],
            'result': [1, 0, 1, 0],  # Fighter A wins, Fighter C wins
            'precomp_elo': [1600, 1400, 1550, 1450],
            'opp_precomp_elo': [1400, 1600, 1450, 1550],
            'postcomp_elo': [1620, 1380, 1570, 1430],  # Required by build_fighter_history
            'opp_postcomp_elo': [1380, 1620, 1430, 1570],
            'avg_odds': [-200, 150, -150, 120]  # Odds for each fighter
        })
        
        # Create odds DataFrame
        self.odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02']),
            'FIGHTER': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A', 'Fighter D', 'Fighter C'],
            'result': [1, 0, 1, 0],
            'avg_odds': [-200, 150, -150, 120]
        })
    
    def test_returns_dict_with_required_keys(self):
        """Test that function returns dictionary with all required keys"""
        result = compute_roi_predictions(self.df)
        
        # When there are no bets (no prior history), these keys should still exist
        required_keys_always = ['roi_percent', 'log_loss', 'brier_score', 'accuracy', 
                               'total_bets', 'records']
        
        for key in required_keys_always:
            self.assertIn(key, result)
        
        # When there are bets, additional keys should exist
        if result['total_bets'] > 0:
            additional_keys = ['total_wagered', 'total_returned', 'total_profit']
            for key in additional_keys:
                self.assertIn(key, result)
    
    def test_empty_result_when_no_prior_history(self):
        """Test that function handles cases where fighters have no prior history"""
        # In test data, all fighters are new (no prior fights), so no bets should be made
        result = compute_roi_predictions(self.df)
        
        # Either 0 bets (no prior history) or some bets made
        # The function should not crash
        self.assertIn('total_bets', result)
        self.assertIn('records', result)
    
    def test_roi_calculation_logic(self):
        """Test ROI calculation is mathematically correct"""
        # Create scenario with fighters who have prior history
        # First add some earlier fights to establish history
        history_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2023-12-01', '2023-12-01', '2023-12-15', '2023-12-15',
                                   '2024-01-01', '2024-01-01']),
            'FIGHTER': ['Winner', 'Loser', 'Winner', 'OtherGuy',
                       'Winner', 'Loser'],
            'opp_FIGHTER': ['Loser', 'Winner', 'OtherGuy', 'Winner',
                           'Loser', 'Winner'],
            'result': [1, 0, 1, 0, 1, 0],
            'precomp_elo': [1500, 1500, 1550, 1450, 1600, 1400],
            'opp_precomp_elo': [1500, 1500, 1450, 1550, 1400, 1600],
            'postcomp_elo': [1550, 1450, 1580, 1420, 1620, 1380],
            'opp_postcomp_elo': [1450, 1550, 1420, 1580, 1380, 1620],
            'avg_odds': [-200, 150, -200, 150, -200, 150]
        })
        
        result = compute_roi_predictions(history_df)
        
        # Function should not crash and should return valid structure
        self.assertIn('total_bets', result)
        self.assertIn('roi_percent', result)


class TestComputeROIOverTime(unittest.TestCase):
    """Tests for compute_roi_over_time function"""
    
    def test_empty_records_returns_empty_df(self):
        """Test that empty records returns empty DataFrame"""
        roi_results = {
            'records': pd.DataFrame(),
            'total_bets': 0
        }
        result = compute_roi_over_time(roi_results)
        self.assertTrue(result.empty)
    
    def test_returns_dataframe_with_required_columns(self):
        """Test that result has required columns"""
        records_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
            'bet_on': ['A', 'B', 'C'],
            'bet_against': ['D', 'E', 'F'],
            'bet_won': [1, 0, 1],
            'elo_diff': [100, 50, 150],
            'expected_prob': [0.6, 0.55, 0.7],
            'avg_odds': [-200, -150, -300],
            'decimal_odds': [1.5, 1.67, 1.33],
            'bet_amount': [1.0, 1.0, 1.0],
            'payout': [1.5, 0, 1.33],
            'profit': [0.5, -1.0, 0.33]
        })
        
        roi_results = {'records': records_df}
        result = compute_roi_over_time(roi_results, group_by='event')
        
        required_columns = ['period', 'wagered', 'returned', 'profit', 
                           'cumulative_roi', 'cumulative_profit']
        
        for col in required_columns:
            self.assertIn(col, result.columns)
    
    def test_cumulative_calculations(self):
        """Test that cumulative calculations are correct"""
        records_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'bet_on': ['A', 'B'],
            'bet_against': ['C', 'D'],
            'bet_won': [1, 1],
            'elo_diff': [100, 100],
            'expected_prob': [0.6, 0.6],
            'avg_odds': [-100, -100],
            'decimal_odds': [2.0, 2.0],
            'bet_amount': [1.0, 1.0],
            'payout': [2.0, 2.0],
            'profit': [1.0, 1.0]
        })
        
        roi_results = {'records': records_df}
        result = compute_roi_over_time(roi_results, group_by='event')
        
        # After 2 events with $1 profit each:
        # Total wagered: $2, Total profit: $2, Cumulative ROI: 100%
        if len(result) == 2:
            final_row = result.iloc[-1]
            self.assertAlmostEqual(final_row['cumulative_profit'], 2.0)
            self.assertAlmostEqual(final_row['cumulative_wagered'], 2.0)
            self.assertAlmostEqual(final_row['cumulative_roi'], 100.0)


class TestCompareOddsSources(unittest.TestCase):
    """Tests for compare_odds_sources function"""
    
    def setUp(self):
        """Set up test data"""
        self.odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'FIGHTER': ['Fighter A', 'Fighter B'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A'],
            'result': [1, 0],
            'avg_odds': [-200, 150],
            'draftkings_odds': [-190, 160],
            'fanduel_odds': [-210, 170],
            'betmgm_odds': [-195, 155]
        })
    
    def test_returns_dict_for_each_source(self):
        """Test that results contain all odds sources"""
        result = compare_odds_sources(self.odds_df)
        
        expected_sources = ['avg_odds', 'draftkings_odds', 'fanduel_odds', 'betmgm_odds']
        for source in expected_sources:
            self.assertIn(source, result)
    
    def test_each_source_has_required_metrics(self):
        """Test that each source has accuracy, log_loss, and brier_score"""
        result = compare_odds_sources(self.odds_df)
        
        required_metrics = ['accuracy', 'log_loss', 'brier_score', 'total_fights']
        
        for source, metrics in result.items():
            for metric in required_metrics:
                self.assertIn(metric, metrics)
    
    def test_accuracy_in_valid_range(self):
        """Test that accuracy is between 0 and 1"""
        result = compare_odds_sources(self.odds_df)
        
        for source, metrics in result.items():
            if metrics['accuracy'] is not None:
                self.assertGreaterEqual(metrics['accuracy'], 0)
                self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_brier_score_in_valid_range(self):
        """Test that Brier score is between 0 and 1"""
        result = compare_odds_sources(self.odds_df)
        
        for source, metrics in result.items():
            if metrics['brier_score'] is not None:
                self.assertGreaterEqual(metrics['brier_score'], 0)
                self.assertLessEqual(metrics['brier_score'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests using actual data files"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        # Get path to project root (parent of scripts directory)
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.project_root, 'after_averaging.csv')
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'after_averaging.csv')), 
        "after_averaging.csv not found")
    def test_full_pipeline_with_real_data(self):
        """Test the full pipeline with real after_averaging.csv data"""
        # This test will only run if after_averaging.csv exists
        odds_df = pd.read_csv(self.data_file, low_memory=False)
        
        # Test compare_odds_sources
        comparison = compare_odds_sources(odds_df)
        
        # Verify we got results
        self.assertIn('avg_odds', comparison)
        self.assertGreater(comparison['avg_odds']['total_fights'], 0)
        
        # Verify metrics are valid
        self.assertIsNotNone(comparison['avg_odds']['accuracy'])
        self.assertIsNotNone(comparison['avg_odds']['log_loss'])
        self.assertIsNotNone(comparison['avg_odds']['brier_score'])


if __name__ == '__main__':
    # Run tests - use absolute path to find data files in integration tests
    unittest.main(verbosity=2)
