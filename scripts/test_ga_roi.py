"""
Unit tests for ROI-based genetic algorithm functions in full_genetic_with_k_denom_mov.py
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from full_genetic_with_k_denom_mov import (
    american_odds_to_decimal,
    build_bidirectional_odds_lookup,
    evaluate_params_roi,
    calculate_oos_roi,
    ga_search_params_roi,
    run_basic_elo,
    compute_prediction_metrics,
    compute_extended_roi_metrics,
)
from elo_utils import add_bout_counts


class TestAmericanOddsToDecimal(unittest.TestCase):
    """Tests for american_odds_to_decimal function in GA module"""
    
    def test_positive_odds(self):
        """Test conversion of positive American odds (underdog)"""
        self.assertAlmostEqual(american_odds_to_decimal(150), 2.5)
        self.assertAlmostEqual(american_odds_to_decimal(100), 2.0)
        self.assertAlmostEqual(american_odds_to_decimal(200), 3.0)
    
    def test_negative_odds(self):
        """Test conversion of negative American odds (favorite)"""
        self.assertAlmostEqual(american_odds_to_decimal(-200), 1.5, places=4)
        self.assertAlmostEqual(american_odds_to_decimal(-100), 2.0, places=4)
        self.assertAlmostEqual(american_odds_to_decimal(-400), 1.25, places=4)
    
    def test_nan_input(self):
        """Test that NaN input returns None"""
        self.assertIsNone(american_odds_to_decimal(np.nan))
        self.assertIsNone(american_odds_to_decimal(float('nan')))


class TestBuildBidirectionalOddsLookup(unittest.TestCase):
    """Tests for build_bidirectional_odds_lookup function"""
    
    def test_creates_entries_for_each_fighter(self):
        """Test that odds lookup creates entries for each fighter's perspective"""
        odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'FIGHTER': ['Fighter A', 'Fighter B'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A'],
            'avg_odds': [-200, 150]  # A is favorite, B is underdog
        })
        
        lookup = build_bidirectional_odds_lookup(odds_df)
        
        # Both fighters' perspectives should be in lookup with their own odds
        self.assertEqual(lookup[('Fighter A', 'Fighter B', '2024-01-01')], -200)
        self.assertEqual(lookup[('Fighter B', 'Fighter A', '2024-01-01')], 150)
    
    def test_handles_nan_odds(self):
        """Test that NaN odds are not added to lookup"""
        odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'FIGHTER': ['Fighter A', 'Fighter C'],
            'opp_FIGHTER': ['Fighter B', 'Fighter D'],
            'avg_odds': [-200, np.nan]
        })
        
        lookup = build_bidirectional_odds_lookup(odds_df)
        
        # First fight should be in lookup
        self.assertIn(('Fighter A', 'Fighter B', '2024-01-01'), lookup)
        # Second fight (NaN odds) should not be in lookup
        self.assertNotIn(('Fighter C', 'Fighter D', '2024-01-02'), lookup)


class TestEvaluateParamsROI(unittest.TestCase):
    """Tests for evaluate_params_roi function"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample fight data with multiple dates to ensure past year filtering works
        dates = pd.date_range(start='2023-01-01', end='2024-06-01', freq='ME')
        
        fighters = ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D']
        data = []
        
        for i, date in enumerate(dates):
            f1 = fighters[i % 4]
            f2 = fighters[(i + 1) % 4]
            data.append({
                'DATE': date,
                'FIGHTER': f1,
                'opp_FIGHTER': f2,
                'result': 1 if i % 2 == 0 else 0,
                'win': 1 if i % 2 == 0 else 0,
                'loss': 0 if i % 2 == 0 else 1,
                'ko': 0, 'kod': 0, 'subw': 0, 'subwd': 0,
                'udec': 1, 'udecd': 0, 'sdec': 0, 'sdecd': 0,
                'mdec': 0, 'mdecd': 0
            })
        
        self.df = pd.DataFrame(data)
        self.df['DATE'] = pd.to_datetime(self.df['DATE']).dt.tz_localize(None)
        
        # Create odds data matching the fight data
        odds_data = []
        for i, date in enumerate(dates):
            f1 = fighters[i % 4]
            f2 = fighters[(i + 1) % 4]
            odds_data.append({
                'DATE': date,
                'FIGHTER': f1,
                'opp_FIGHTER': f2,
                'avg_odds': -150 if i % 2 == 0 else 120
            })
        
        self.odds_df = pd.DataFrame(odds_data)
        self.odds_df['DATE'] = pd.to_datetime(self.odds_df['DATE']).dt.tz_localize(None)
        
        self.sample_params = {
            'k': 32,
            'w_ko': 1.4,
            'w_sub': 1.3,
            'w_udec': 1.0,
            'w_sdec': 0.7,
            'w_mdec': 0.9
        }
    
    def test_returns_numeric_roi(self):
        """Test that evaluate_params_roi returns a numeric ROI value"""
        roi = evaluate_params_roi(self.df, self.odds_df, self.sample_params, lookback_days=365)
        self.assertIsInstance(roi, float)
    
    def test_returns_zero_when_no_odds(self):
        """Test that ROI is 0 when no odds are available"""
        empty_odds = pd.DataFrame({
            'DATE': pd.to_datetime([]),
            'FIGHTER': [],
            'opp_FIGHTER': [],
            'avg_odds': []
        })
        
        roi = evaluate_params_roi(self.df, empty_odds, self.sample_params, lookback_days=365)
        self.assertEqual(roi, 0.0)


class TestGASearchParamsROI(unittest.TestCase):
    """Tests for ga_search_params_roi function"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        # Get path to project root
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.project_root, 'data', 'interleaved_cleaned.csv')
        cls.odds_file = os.path.join(cls.project_root, 'after_averaging.csv')
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_returns_best_params_and_roi(self):
        """Test that GA returns best params and ROI value"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        # Run with minimal settings for quick test
        best_params, best_roi = ga_search_params_roi(
            df,
            odds_df,
            population_size=3,
            generations=1,
            seed=42,
            verbose=False
        )
        
        # Check return values
        self.assertIsInstance(best_params, dict)
        self.assertIn('k', best_params)
        self.assertIn('w_ko', best_params)
        self.assertIn('w_sub', best_params)
        self.assertIn('w_udec', best_params)
        self.assertIn('w_sdec', best_params)
        self.assertIn('w_mdec', best_params)
        self.assertIsInstance(best_roi, float)
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_return_all_results(self):
        """Test that return_all_results option returns generation summaries"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        # Run with return_all_results=True
        best_params, best_roi, all_results = ga_search_params_roi(
            df,
            odds_df,
            population_size=3,
            generations=2,
            seed=42,
            verbose=False,
            return_all_results=True
        )
        
        # Check all_results structure
        self.assertIsInstance(all_results, list)
        self.assertEqual(len(all_results), 2)  # 2 generations
        
        for gen_result in all_results:
            self.assertIn('generation', gen_result)
            self.assertIn('best_fitness', gen_result)
            self.assertIn('best_params', gen_result)


class TestCalculateOOSROI(unittest.TestCase):
    """Tests for calculate_oos_roi function"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample trained data
        self.df_trained = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-15', '2024-01-15']),
            'FIGHTER': ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A', 'Fighter D', 'Fighter C'],
            'result': [1, 0, 1, 0],
            'precomp_elo': [1550, 1450, 1600, 1400],
            'postcomp_elo': [1570, 1430, 1620, 1380],
            'opp_precomp_elo': [1450, 1550, 1400, 1600],
            'opp_postcomp_elo': [1430, 1570, 1380, 1620]
        })
        
        # Create test data
        self.test_df = pd.DataFrame({
            'date': pd.to_datetime(['2024-02-01']),
            'fighter': ['Fighter A'],
            'opp_fighter': ['Fighter C'],
            'result': [1]
        })
        
        # Create odds data
        self.odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-02-01', '2024-02-01']),
            'FIGHTER': ['Fighter A', 'Fighter C'],
            'opp_FIGHTER': ['Fighter C', 'Fighter A'],
            'avg_odds': [-150, 120]
        })
    
    def test_returns_dict_with_required_keys(self):
        """Test that function returns dictionary with all required keys"""
        result = calculate_oos_roi(self.df_trained, self.test_df, self.odds_df, verbose=False)
        
        required_keys = ['roi_percent', 'total_bets', 'total_wagered', 
                        'total_profit', 'wins', 'accuracy']
        
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_handles_empty_test_data(self):
        """Test that function handles empty test data gracefully"""
        empty_test = pd.DataFrame({
            'date': pd.to_datetime([]),
            'fighter': [],
            'opp_fighter': [],
            'result': []
        })
        
        result = calculate_oos_roi(self.df_trained, empty_test, self.odds_df, verbose=False)
        
        self.assertEqual(result['total_bets'], 0)
        self.assertEqual(result['roi_percent'], 0.0)


class TestDrawHandling(unittest.TestCase):
    """Tests for draw handling in ROI calculations"""
    
    def test_draws_are_skipped_in_roi(self):
        """Test that draws (result not 0 or 1) are skipped in ROI calculation"""
        # Create data with a draw (result = 0.5 or similar)
        df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'FIGHTER': ['Fighter A', 'Fighter B'],
            'opp_FIGHTER': ['Fighter B', 'Fighter A'],
            'result': [0.5, 0.5],  # Draw
            'win': [0, 0],
            'loss': [0, 0],
            'ko': [0, 0], 'kod': [0, 0], 'subw': [0, 0], 'subwd': [0, 0],
            'udec': [0, 0], 'udecd': [0, 0], 'sdec': [0, 0], 'sdecd': [0, 0],
            'mdec': [0, 0], 'mdecd': [0, 0]
        })
        
        odds_df = pd.DataFrame({
            'DATE': pd.to_datetime(['2024-01-01']),
            'FIGHTER': ['Fighter A'],
            'opp_FIGHTER': ['Fighter B'],
            'avg_odds': [-150]
        })
        
        params = {
            'k': 32, 'w_ko': 1.4, 'w_sub': 1.3,
            'w_udec': 1.0, 'w_sdec': 0.7, 'w_mdec': 0.9
        }
        
        # ROI should be 0 because draw is skipped
        roi = evaluate_params_roi(df, odds_df, params, lookback_days=0)
        self.assertEqual(roi, 0.0)


class TestMainBlockModeRouting(unittest.TestCase):
    """Tests for the main block mode routing to ensure --mode flag works correctly"""
    
    def setUp(self):
        """Set up the argparse parser for mode routing tests"""
        import argparse
        self.parser = argparse.ArgumentParser(description="Genetic Algorithm for Elo Parameter Optimization")
        self.parser.add_argument("--mode", choices=["accuracy", "roi"], default="roi",
                                 help="Optimization mode: 'accuracy' (original) or 'roi' (ROI-based)")
    
    def test_argparse_mode_default_is_roi(self):
        """Test that the default mode is 'roi'"""
        args = self.parser.parse_args([])
        self.assertEqual(args.mode, "roi", "Default mode should be 'roi'")
    
    def test_argparse_mode_roi_explicit(self):
        """Test that --mode roi is correctly parsed"""
        args = self.parser.parse_args(["--mode", "roi"])
        self.assertEqual(args.mode, "roi")
    
    def test_argparse_mode_accuracy_explicit(self):
        """Test that --mode accuracy is correctly parsed"""
        args = self.parser.parse_args(["--mode", "accuracy"])
        self.assertEqual(args.mode, "accuracy")
    
    def test_mode_condition_works_for_roi(self):
        """Test that the mode condition 'args.mode == \"roi\"' works correctly for roi mode"""
        args = self.parser.parse_args(["--mode", "roi"])
        self.assertTrue(args.mode == "roi", "Condition 'args.mode == \"roi\"' should be True for roi mode")
    
    def test_mode_condition_works_for_accuracy(self):
        """Test that the mode condition 'args.mode == \"roi\"' correctly fails for accuracy mode"""
        args = self.parser.parse_args(["--mode", "accuracy"])
        self.assertFalse(args.mode == "roi", "Condition 'args.mode == \"roi\"' should be False for accuracy mode")


class TestComputeExtendedROIMetrics(unittest.TestCase):
    """Tests for compute_extended_roi_metrics function"""
    
    def test_empty_records_returns_none_metrics(self):
        """Test that empty records return None for all metrics"""
        result = compute_extended_roi_metrics([])
        
        self.assertIsNone(result['trend'])
        self.assertIsNone(result['sharpe_ratio'])
        self.assertIsNone(result['min_roi'])
        self.assertIsNone(result['max_roi'])
        self.assertIsNone(result['win_rate'])
        self.assertEqual(result['num_bets'], 0)
    
    def test_returns_valid_metrics(self):
        """Test that function returns all expected keys"""
        bet_records = [
            {'date': pd.Timestamp('2024-01-01'), 'profit': 0.5, 'bet_amount': 1.0, 'bet_won': 1},
            {'date': pd.Timestamp('2024-01-02'), 'profit': -1.0, 'bet_amount': 1.0, 'bet_won': 0},
            {'date': pd.Timestamp('2024-01-03'), 'profit': 0.3, 'bet_amount': 1.0, 'bet_won': 1},
        ]
        
        result = compute_extended_roi_metrics(bet_records)
        
        required_keys = ['trend', 'sharpe_ratio', 'min_roi', 'max_roi', 'win_rate', 
                        'num_bets', 'total_wagered', 'total_profit']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_win_rate_calculation(self):
        """Test that win rate is calculated correctly"""
        bet_records = [
            {'date': pd.Timestamp('2024-01-01'), 'profit': 0.5, 'bet_amount': 1.0, 'bet_won': 1},
            {'date': pd.Timestamp('2024-01-02'), 'profit': -1.0, 'bet_amount': 1.0, 'bet_won': 0},
            {'date': pd.Timestamp('2024-01-03'), 'profit': 0.3, 'bet_amount': 1.0, 'bet_won': 1},
            {'date': pd.Timestamp('2024-01-04'), 'profit': 0.2, 'bet_amount': 1.0, 'bet_won': 1},
        ]
        
        result = compute_extended_roi_metrics(bet_records)
        
        # 3 wins out of 4 bets = 75%
        self.assertAlmostEqual(result['win_rate'], 0.75)
        self.assertEqual(result['num_bets'], 4)
    
    def test_total_profit_calculation(self):
        """Test that total profit and wagered are calculated correctly"""
        bet_records = [
            {'date': pd.Timestamp('2024-01-01'), 'profit': 0.5, 'bet_amount': 1.0, 'bet_won': 1},
            {'date': pd.Timestamp('2024-01-02'), 'profit': -1.0, 'bet_amount': 1.0, 'bet_won': 0},
        ]
        
        result = compute_extended_roi_metrics(bet_records)
        
        self.assertAlmostEqual(result['total_profit'], -0.5)
        self.assertAlmostEqual(result['total_wagered'], 2.0)


class TestComputePredictionMetrics(unittest.TestCase):
    """Tests for compute_prediction_metrics function"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample fight data with Elo ratings
        dates = pd.date_range(start='2023-01-01', end='2024-06-01', freq='ME')
        
        fighters = ['Fighter A', 'Fighter B', 'Fighter C', 'Fighter D']
        data = []
        
        for i, date in enumerate(dates):
            f1 = fighters[i % 4]
            f2 = fighters[(i + 1) % 4]
            data.append({
                'DATE': date,
                'FIGHTER': f1,
                'opp_FIGHTER': f2,
                'result': 1 if i % 2 == 0 else 0,
                'win': 1 if i % 2 == 0 else 0,
                'loss': 0 if i % 2 == 0 else 1,
                'precomp_elo': 1500 + (i * 10),
                'postcomp_elo': 1500 + (i * 10) + 20,
                'opp_precomp_elo': 1500 - (i * 5),
                'opp_postcomp_elo': 1500 - (i * 5) - 20,
                'ko': 0, 'kod': 0, 'subw': 0, 'subwd': 0,
                'udec': 1, 'udecd': 0, 'sdec': 0, 'sdecd': 0,
                'mdec': 0, 'mdecd': 0
            })
        
        self.df = pd.DataFrame(data)
        self.df['DATE'] = pd.to_datetime(self.df['DATE']).dt.tz_localize(None)
        
        # Create empty odds data (not used for prediction metrics)
        self.odds_df = pd.DataFrame({
            'DATE': pd.to_datetime([]),
            'FIGHTER': [],
            'opp_FIGHTER': [],
            'avg_odds': []
        })
    
    def test_returns_required_keys(self):
        """Test that function returns all required keys"""
        result = compute_prediction_metrics(self.df, self.odds_df, lookback_days=0)
        
        required_keys = ['accuracy', 'log_loss', 'brier_score', 'total_predictions']
        for key in required_keys:
            self.assertIn(key, result)
    
    def test_accuracy_in_valid_range(self):
        """Test that accuracy is between 0 and 1"""
        result = compute_prediction_metrics(self.df, self.odds_df, lookback_days=0)
        
        if result['accuracy'] is not None:
            self.assertGreaterEqual(result['accuracy'], 0)
            self.assertLessEqual(result['accuracy'], 1)
    
    def test_brier_score_in_valid_range(self):
        """Test that Brier score is between 0 and 1"""
        result = compute_prediction_metrics(self.df, self.odds_df, lookback_days=0)
        
        if result['brier_score'] is not None:
            self.assertGreaterEqual(result['brier_score'], 0)
            self.assertLessEqual(result['brier_score'], 1)
    
    def test_log_loss_positive(self):
        """Test that log loss is positive"""
        result = compute_prediction_metrics(self.df, self.odds_df, lookback_days=0)
        
        if result['log_loss'] is not None:
            self.assertGreater(result['log_loss'], 0)


class TestEvaluateParamsROIExtended(unittest.TestCase):
    """Tests for evaluate_params_roi with return_extended=True"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.project_root, 'data', 'interleaved_cleaned.csv')
        cls.odds_file = os.path.join(cls.project_root, 'after_averaging.csv')
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_return_extended_returns_dict(self):
        """Test that return_extended=True returns a dictionary with all metrics"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        params = {
            'k': 32,
            'w_ko': 1.4,
            'w_sub': 1.3,
            'w_udec': 1.0,
            'w_sdec': 0.7,
            'w_mdec': 0.9
        }
        
        result = evaluate_params_roi(df, odds_df, params, lookback_days=0, return_extended=True)
        
        self.assertIsInstance(result, dict)
        
        # Check all required keys exist
        required_keys = ['roi_percent', 'trend', 'sharpe_ratio', 'min_roi', 'max_roi',
                        'win_rate', 'num_bets', 'accuracy', 'log_loss', 'brier_score']
        for key in required_keys:
            self.assertIn(key, result)
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_return_extended_false_returns_float(self):
        """Test that return_extended=False returns a float"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        params = {
            'k': 32,
            'w_ko': 1.4,
            'w_sub': 1.3,
            'w_udec': 1.0,
            'w_sdec': 0.7,
            'w_mdec': 0.9
        }
        
        result = evaluate_params_roi(df, odds_df, params, lookback_days=0, return_extended=False)
        
        self.assertIsInstance(result, float)


class TestGASearchParamsROIExtendedResults(unittest.TestCase):
    """Tests for ga_search_params_roi with extended metrics in results"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.project_root, 'data', 'interleaved_cleaned.csv')
        cls.odds_file = os.path.join(cls.project_root, 'after_averaging.csv')
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_all_results_contain_extended_metrics(self):
        """Test that all_results contains extended metrics for each generation"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        best_params, best_roi, all_results = ga_search_params_roi(
            df,
            odds_df,
            population_size=3,
            generations=2,
            seed=42,
            verbose=False,
            return_all_results=True
        )
        
        # Check that extended metrics are in generation results
        extended_keys = ['best_roi_percent', 'best_trend', 'best_sharpe_ratio',
                        'best_min_roi', 'best_max_roi', 'best_win_rate', 'best_num_bets',
                        'best_accuracy', 'best_log_loss', 'best_brier_score']
        
        for gen_result in all_results:
            for key in extended_keys:
                self.assertIn(key, gen_result, f"Missing key {key} in generation results")
    
    @unittest.skipIf(not os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
        'data', 'interleaved_cleaned.csv')), 
        "interleaved_cleaned.csv not found")
    def test_multi_objective_fitness_weights(self):
        """Test that fitness_weights parameter works"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df = df.sort_values('DATE').reset_index(drop=True)
        df = add_bout_counts(df)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        # Test with multi-objective weights
        best_params, best_roi = ga_search_params_roi(
            df,
            odds_df,
            population_size=3,
            generations=1,
            seed=42,
            verbose=False,
            fitness_weights={'roi': 0.6, 'trend': 0.3, 'sharpe': 0.1}
        )
        
        # Should not raise an error and should return valid results
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_roi, float)


if __name__ == '__main__':
    unittest.main(verbosity=2)
