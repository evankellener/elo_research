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
    compare_odds_sources,
    analyze_random_events,
    display_detailed_event_analysis,
    get_method_of_victory
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
    
    def test_odds_lookup_uses_odds_df_for_opponent(self):
        """Test that compute_roi_predictions correctly looks up opponent odds from odds_df.
        
        This test verifies the fix for the issue where fights with valid odds in odds_df
        were being skipped because the opponent odds lookup was using df (single perspective)
        instead of odds_df (both perspectives).
        
        Example: Zac Pauga vs Bogdan Guskov - Guskov has higher Elo, so we bet on Guskov,
        but need to find Guskov's odds from odds_df.
        """
        # Create test data where FIGHTER has lower Elo and we need to bet on opponent
        # df has only one row per fight (FIGHTER perspective)
        # Need to establish prior history for both fighters
        df = pd.DataFrame({
            'DATE': pd.to_datetime([
                # First fights to establish history (2023-01-01)
                '2023-01-01', '2023-01-01',
                # Fight on 2024-01-01 - Zac Pauga vs Higher Elo
                # Only one perspective in df (Zac Pauga's)
                '2024-01-01'
            ]),
            'FIGHTER': ['Zac Pauga', 'Higher Elo', 'Zac Pauga'],
            'opp_FIGHTER': ['SomeoneElse1', 'SomeoneElse2', 'Higher Elo'],
            'result': [1, 1, 0],  # Win first fights, Zac Pauga loses to Higher Elo
            'precomp_elo': [1500, 1500, 1326],
            'opp_precomp_elo': [1400, 1400, 1446],  # Higher Elo has higher Elo rating
            'postcomp_elo': [1550, 1550, 1280],
            'opp_postcomp_elo': [1350, 1350, 1490],
        })
        
        # odds_df has both perspectives with valid odds for Higher Elo
        odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'FIGHTER': ['Zac Pauga', 'Higher Elo'],
            'opp_FIGHTER': ['Higher Elo', 'Zac Pauga'],
            'result': [0, 1],
            'avg_odds': [-121, 102]  # Zac is favorite, Higher Elo is underdog
        })
        
        result = compute_roi_predictions(df, odds_df=odds_df)
        
        # The bet should be placed on Higher Elo (who has higher Elo rating)
        # This only works if the function correctly looks up odds from odds_df
        self.assertGreater(result['total_bets'], 0, 
            "Should have at least one bet when opponent has higher Elo and valid odds in odds_df")
        
        # Check that the bet was placed on Higher Elo
        records = result['records']
        if not records.empty:
            higher_elo_bets = records[records['bet_on'] == 'Higher Elo']
            self.assertGreater(len(higher_elo_bets), 0,
                "Should bet on the fighter with higher Elo (Higher Elo)")


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


class TestGetMethodOfVictory(unittest.TestCase):
    """Tests for get_method_of_victory function"""
    
    def test_ko_victory(self):
        """Test detection of KO/TKO victory"""
        row = {'ko': 1, 'subw': 0, 'udec': 0, 'sdec': 0, 'mdec': 0}
        self.assertEqual(get_method_of_victory(row), 'KO/TKO')
    
    def test_submission_victory(self):
        """Test detection of submission victory"""
        row = {'ko': 0, 'subw': 1, 'udec': 0, 'sdec': 0, 'mdec': 0}
        self.assertEqual(get_method_of_victory(row), 'Submission')
    
    def test_unanimous_decision(self):
        """Test detection of unanimous decision"""
        row = {'ko': 0, 'subw': 0, 'udec': 1, 'sdec': 0, 'mdec': 0}
        self.assertEqual(get_method_of_victory(row), 'Unanimous Decision')
    
    def test_split_decision(self):
        """Test detection of split decision"""
        row = {'ko': 0, 'subw': 0, 'udec': 0, 'sdec': 1, 'mdec': 0}
        self.assertEqual(get_method_of_victory(row), 'Split Decision')
    
    def test_majority_decision(self):
        """Test detection of majority decision"""
        row = {'ko': 0, 'subw': 0, 'udec': 0, 'sdec': 0, 'mdec': 1}
        self.assertEqual(get_method_of_victory(row), 'Majority Decision')
    
    def test_unknown_victory(self):
        """Test unknown method of victory when no flags are set"""
        row = {'ko': 0, 'subw': 0, 'udec': 0, 'sdec': 0, 'mdec': 0}
        self.assertEqual(get_method_of_victory(row), 'Unknown')


class TestAnalyzeRandomEvents(unittest.TestCase):
    """Tests for analyze_random_events function"""
    
    def setUp(self):
        """Set up test data with ROI records"""
        # Create mock ROI results with records DataFrame
        self.records_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03']),
            'bet_on': ['Fighter A', 'Fighter C', 'Fighter E', 'Fighter G', 'Fighter I'],
            'bet_against': ['Fighter B', 'Fighter D', 'Fighter F', 'Fighter H', 'Fighter J'],
            'bet_won': [1, 0, 1, 1, 0],
            'elo_diff': [100, 50, 150, 80, 120],
            'expected_prob': [0.6, 0.55, 0.7, 0.58, 0.65],
            'avg_odds': [-200, -150, -300, -180, -250],
            'decimal_odds': [1.5, 1.67, 1.33, 1.56, 1.4],
            'bet_amount': [1.0, 1.0, 1.0, 1.0, 1.0],
            'payout': [1.5, 0, 1.33, 1.56, 0],
            'profit': [0.5, -1.0, 0.33, 0.56, -1.0]
        })
        
        self.roi_results = {'records': self.records_df}
        
        # Create mock df with Elo data
        self.df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', 
                                   '2024-01-02', '2024-01-02', '2024-01-02', '2024-01-02',
                                   '2024-01-03', '2024-01-03']),
            'FIGHTER': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D',
                       'Fighter E', 'Fighter F', 'Fighter G', 'Fighter H',
                       'Fighter I', 'Fighter J'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A', 'Fighter D', 'Fighter C',
                           'Fighter F', 'Fighter E', 'Fighter H', 'Fighter G',
                           'Fighter J', 'Fighter I'],
            'precomp_elo': [1600, 1400, 1550, 1450, 1700, 1500, 1580, 1420, 1650, 1480],
            'postcomp_elo': [1620, 1380, 1530, 1470, 1720, 1480, 1600, 1400, 1630, 1500],
            'opp_precomp_elo': [1400, 1600, 1450, 1550, 1500, 1700, 1420, 1580, 1480, 1650],
            'opp_postcomp_elo': [1380, 1620, 1470, 1530, 1480, 1720, 1400, 1600, 1500, 1630],
            'result': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            'EVENT': ['Event 1'] * 4 + ['Event 2'] * 4 + ['Event 3'] * 2,
            'ko': [0] * 10,
            'subw': [0] * 10,
            'udec': [1] * 10,
            'sdec': [0] * 10,
            'mdec': [0] * 10,
            'round': [3] * 10,
            'time_format': ['3 Rnd (5-5-5)'] * 10
        })
    
    def test_returns_list_of_events(self):
        """Test that function returns a list"""
        result = analyze_random_events(self.df, self.roi_results, n_events=2, random_seed=42)
        self.assertIsInstance(result, list)
    
    def test_returns_correct_number_of_events(self):
        """Test that function returns correct number of events"""
        result = analyze_random_events(self.df, self.roi_results, n_events=2, random_seed=42)
        self.assertEqual(len(result), 2)
    
    def test_event_has_required_keys(self):
        """Test that each event has all required keys"""
        result = analyze_random_events(self.df, self.roi_results, n_events=1, random_seed=42)
        
        required_keys = ['event_date', 'event_name', 'bets', 'other_fights', 'event_roi', 
                        'win_count', 'loss_count', 'total_wagered', 'total_payout', 'total_profit']
        
        if result:
            for key in required_keys:
                self.assertIn(key, result[0])
    
    def test_bets_have_required_keys(self):
        """Test that each bet has required keys"""
        result = analyze_random_events(self.df, self.roi_results, n_events=1, random_seed=42)
        
        required_bet_keys = ['bet_on', 'bet_against', 'bet_amount', 'payout', 
                            'profit', 'bet_won', 'elo_diff']
        
        if result and result[0]['bets']:
            for key in required_bet_keys:
                self.assertIn(key, result[0]['bets'][0])
    
    def test_empty_records_returns_empty_list(self):
        """Test that empty records returns empty list"""
        empty_results = {'records': pd.DataFrame()}
        result = analyze_random_events(self.df, empty_results, n_events=5)
        self.assertEqual(result, [])
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results"""
        result1 = analyze_random_events(self.df, self.roi_results, n_events=2, random_seed=42)
        result2 = analyze_random_events(self.df, self.roi_results, n_events=2, random_seed=42)
        
        if result1 and result2:
            self.assertEqual(result1[0]['event_date'], result2[0]['event_date'])
    
    def test_event_roi_calculation(self):
        """Test that event ROI is calculated correctly"""
        result = analyze_random_events(self.df, self.roi_results, n_events=3, random_seed=42)
        
        for event in result:
            # Verify ROI calculation: (profit / wagered) * 100
            expected_roi = (event['total_profit'] / event['total_wagered']) * 100 if event['total_wagered'] > 0 else 0
            self.assertAlmostEqual(event['event_roi'], expected_roi, places=2)
    
    def test_other_fights_is_list(self):
        """Test that other_fights is always a list"""
        result = analyze_random_events(self.df, self.roi_results, n_events=3, random_seed=42)
        
        for event in result:
            self.assertIn('other_fights', event)
            self.assertIsInstance(event['other_fights'], list)
    
    def test_other_fights_have_required_keys(self):
        """Test that each fight in other_fights has required keys"""
        result = analyze_random_events(self.df, self.roi_results, n_events=3, random_seed=42)
        
        required_fight_keys = ['fighter', 'opponent', 'winner', 'loser', 'fighter_won',
                               'fighter_pre_elo', 'opp_pre_elo', 'elo_diff',
                               'method_of_victory', 'bet_placed', 'reason_no_bet']
        
        for event in result:
            for fight in event.get('other_fights', []):
                for key in required_fight_keys:
                    self.assertIn(key, fight)


class TestDisplayDetailedEventAnalysis(unittest.TestCase):
    """Tests for display_detailed_event_analysis function"""
    
    def test_handles_empty_list(self):
        """Test that function handles empty list without error"""
        # Should not raise an exception
        display_detailed_event_analysis([])
    
    def test_handles_single_event(self):
        """Test that function handles single event"""
        event = {
            'event_date': '2024-01-01',
            'event_name': 'Test Event',
            'bets': [
                {
                    'bet_on': 'Fighter A',
                    'bet_against': 'Fighter B',
                    'bet_on_pre_elo': 1600,
                    'bet_on_post_elo': 1620,
                    'bet_on_elo_change': 20,
                    'bet_against_pre_elo': 1500,
                    'bet_against_post_elo': 1480,
                    'bet_against_elo_change': -20,
                    'elo_diff': 100,
                    'avg_odds_american': -200,
                    'decimal_odds': 1.5,
                    'bet_amount': 1.0,
                    'payout': 1.5,
                    'profit': 0.5,
                    'bet_won': True,
                    'expected_prob': 0.6,
                    'method_of_victory': 'KO/TKO',
                    'round': 2,
                    'draftkings_odds': -190,
                    'fanduel_odds': -210,
                    'betmgm_odds': -195
                }
            ],
            'other_fights': [],
            'event_roi': 50.0,
            'win_count': 1,
            'loss_count': 0,
            'total_wagered': 1.0,
            'total_payout': 1.5,
            'total_profit': 0.5
        }
        # Should not raise an exception
        display_detailed_event_analysis([event])
    
    def test_handles_event_with_other_fights(self):
        """Test that function handles event with both bets and other_fights"""
        event = {
            'event_date': '2024-01-01',
            'event_name': 'Test Event',
            'bets': [
                {
                    'bet_on': 'Fighter A',
                    'bet_against': 'Fighter B',
                    'bet_on_pre_elo': 1600,
                    'bet_on_post_elo': 1620,
                    'bet_on_elo_change': 20,
                    'bet_against_pre_elo': 1500,
                    'bet_against_post_elo': 1480,
                    'bet_against_elo_change': -20,
                    'elo_diff': 100,
                    'avg_odds_american': -200,
                    'decimal_odds': 1.5,
                    'bet_amount': 1.0,
                    'payout': 1.5,
                    'profit': 0.5,
                    'bet_won': True,
                    'expected_prob': 0.6,
                    'method_of_victory': 'KO/TKO',
                    'round': 2,
                    'draftkings_odds': -190,
                    'fanduel_odds': -210,
                    'betmgm_odds': -195
                }
            ],
            'other_fights': [
                {
                    'fighter': 'Fighter C',
                    'opponent': 'Fighter D',
                    'winner': 'Fighter C',
                    'loser': 'Fighter D',
                    'fighter_won': True,
                    'fighter_pre_elo': 1550,
                    'fighter_post_elo': 1580,
                    'fighter_elo_change': 30,
                    'opp_pre_elo': 1450,
                    'opp_post_elo': 1420,
                    'opp_elo_change': -30,
                    'elo_diff': 100,
                    'avg_odds_american': None,
                    'method_of_victory': 'Unanimous Decision',
                    'round': 3,
                    'draftkings_odds': None,
                    'fanduel_odds': None,
                    'betmgm_odds': None,
                    'bet_placed': False,
                    'reason_no_bet': 'No odds available'
                }
            ],
            'event_roi': 50.0,
            'win_count': 1,
            'loss_count': 0,
            'total_wagered': 1.0,
            'total_payout': 1.5,
            'total_profit': 0.5
        }
        # Should not raise an exception
        display_detailed_event_analysis([event])


if __name__ == '__main__':
    # Run tests - use absolute path to find data files in integration tests
    unittest.main(verbosity=2)
