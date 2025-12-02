"""
Unit tests for ga_time_split_roi.py module.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
import shutil

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ga_time_split_roi import (
    compute_split_roi,
    compute_time_split_fitness,
    ga_search_time_split_roi,
    save_results,
)
from time_splitter import create_time_splits

# Module-level path constants
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'interleaved_cleaned.csv')
_ODDS_FILE = os.path.join(_PROJECT_ROOT, 'after_averaging.csv')


class TestComputeTimeSplitFitness(unittest.TestCase):
    """Tests for compute_time_split_fitness function"""
    
    def setUp(self):
        """Create sample data for testing"""
        # Create 2 years of fight data
        dates = pd.date_range('2020-01-01', periods=24, freq='ME')
        
        self.df = pd.DataFrame({
            'DATE': dates,
            'FIGHTER': ['A', 'B', 'C', 'D'] * 6,
            'opp_FIGHTER': ['B', 'A', 'D', 'C'] * 6,
            'result': [1, 0, 1, 0] * 6,
            'win': [1, 0, 1, 0] * 6,
            'loss': [0, 1, 0, 1] * 6,
            'ko': [0] * 24,
            'kod': [0] * 24,
            'subw': [0] * 24,
            'subwd': [0] * 24,
            'udec': [1] * 24,
            'udecd': [0] * 24,
            'sdec': [0] * 24,
            'sdecd': [0] * 24,
            'mdec': [0] * 24,
            'mdecd': [0] * 24,
        })
        
        # Create odds data
        self.odds_df = pd.DataFrame({
            'DATE': dates,
            'FIGHTER': ['A', 'B', 'C', 'D'] * 6,
            'opp_FIGHTER': ['B', 'A', 'D', 'C'] * 6,
            'avg_odds': [-150, 120, -200, 180] * 6,
        })
        
        self.sample_params = {
            'k': 32,
            'w_ko': 1.4,
            'w_sub': 1.3,
            'w_udec': 1.0,
            'w_sdec': 0.7,
            'w_mdec': 0.9,
        }
    
    def test_returns_required_keys(self):
        """Test that function returns all required keys"""
        splits = create_time_splits(self.df, months=6, min_train_months=3)
        
        if len(splits) > 0:
            result = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=1.0, objective="mean_std"
            )
            
            required_keys = ['fitness', 'mean_roi', 'std_roi', 'cv', 
                           'per_split_roi', 'per_split_details']
            for key in required_keys:
                self.assertIn(key, result)
    
    def test_mean_std_objective(self):
        """Test that mean_std objective works correctly"""
        splits = create_time_splits(self.df, months=6, min_train_months=3)
        
        if len(splits) > 0:
            result = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=1.0, objective="mean_std"
            )
            
            # Fitness should be mean_roi - lambda * std_roi
            expected_fitness = result['mean_roi'] - 1.0 * result['std_roi']
            self.assertAlmostEqual(result['fitness'], expected_fitness, places=5)
    
    def test_cv_objective(self):
        """Test that cv (coefficient of variation) objective works"""
        splits = create_time_splits(self.df, months=6, min_train_months=3)
        
        if len(splits) > 0:
            result = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=1.0, objective="cv"
            )
            
            self.assertIn('cv', result)
            self.assertIsInstance(result['fitness'], float)
    
    def test_lambda_affects_fitness(self):
        """Test that lambda parameter affects fitness"""
        splits = create_time_splits(self.df, months=6, min_train_months=3)
        
        if len(splits) > 0:
            result1 = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=0.5, objective="mean_std"
            )
            result2 = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=2.0, objective="mean_std"
            )
            
            # Different lambdas should give different fitness (unless std=0)
            if result1['std_roi'] > 0:
                self.assertNotEqual(result1['fitness'], result2['fitness'])
    
    def test_per_split_roi_count(self):
        """Test that per_split_roi has correct number of values"""
        splits = create_time_splits(self.df, months=6, min_train_months=3)
        
        if len(splits) > 0:
            result = compute_time_split_fitness(
                self.df, self.odds_df, self.sample_params, splits,
                lambda_penalty=1.0, objective="mean_std"
            )
            
            self.assertEqual(len(result['per_split_roi']), len(splits))
            self.assertEqual(len(result['per_split_details']), len(splits))


class TestGASearchTimeSplitROI(unittest.TestCase):
    """Tests for ga_search_time_split_roi function"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        cls.project_root = _PROJECT_ROOT
        cls.data_file = _DATA_FILE
        cls.odds_file = _ODDS_FILE
    
    @unittest.skipIf(not os.path.exists(_DATA_FILE), "Data file not found")
    @unittest.skipIf(not os.path.exists(_ODDS_FILE), "Odds file not found")
    def test_returns_best_params_and_fitness(self):
        """Test that GA returns best params and fitness"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df.sort_values('DATE').reset_index(drop=True)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        # Create splits
        splits = create_time_splits(df, months=12, min_train_months=12)
        
        if len(splits) < 2:
            self.skipTest("Not enough data for time splits")
        
        # Run GA with minimal settings
        best_params, best_fitness, all_results = ga_search_time_split_roi(
            df=df,
            odds_df=odds_df,
            splits=splits[:3],  # Use only first 3 splits for speed
            population_size=3,
            generations=2,
            lambda_penalty=1.0,
            seed=42,
            verbose=False,
            return_all_results=True,
        )
        
        # Check return types
        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_fitness, float)
        self.assertIsInstance(all_results, list)
        
        # Check params structure
        self.assertIn('k', best_params)
        self.assertIn('w_ko', best_params)
        self.assertIn('w_sub', best_params)
    
    @unittest.skipIf(not os.path.exists(_DATA_FILE), "Data file not found")
    @unittest.skipIf(not os.path.exists(_ODDS_FILE), "Odds file not found")
    def test_seed_reproducibility(self):
        """Test that same seed produces same results"""
        df = pd.read_csv(self.data_file, low_memory=False)
        odds_df = pd.read_csv(self.odds_file, low_memory=False)
        
        df['DATE'] = pd.to_datetime(df['DATE']).dt.tz_localize(None)
        df['result'] = pd.to_numeric(df['result'], errors='coerce')
        df = df.sort_values('DATE').reset_index(drop=True)
        
        odds_df['DATE'] = pd.to_datetime(odds_df['DATE']).dt.tz_localize(None)
        
        splits = create_time_splits(df, months=12, min_train_months=12)
        
        if len(splits) < 2:
            self.skipTest("Not enough data for time splits")
        
        # Run twice with same seed
        params1, fitness1, _ = ga_search_time_split_roi(
            df, odds_df, splits[:2],
            population_size=3, generations=1, seed=42, verbose=False,
            return_all_results=True
        )
        
        params2, fitness2, _ = ga_search_time_split_roi(
            df, odds_df, splits[:2],
            population_size=3, generations=1, seed=42, verbose=False,
            return_all_results=True
        )
        
        # Results should be identical
        self.assertEqual(params1, params2)
        self.assertEqual(fitness1, fitness2)


class TestSaveResults(unittest.TestCase):
    """Tests for save_results function"""
    
    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_creates_output_files(self):
        """Test that all expected output files are created"""
        best_params = {
            'k': 32, 'w_ko': 1.4, 'w_sub': 1.3,
            'w_udec': 1.0, 'w_sdec': 0.7, 'w_mdec': 0.9
        }
        extended_result = {
            'mean_roi': 5.0,
            'std_roi': 2.0,
            'cv': 0.4,
            'per_split_roi': [3.0, 5.0, 7.0],
            'per_split_details': [
                {'val_start': pd.Timestamp('2020-01-01'), 'val_end': pd.Timestamp('2020-06-30'),
                 'roi_percent': 3.0, 'num_bets': 10, 'wins': 6, 'total_wagered': 10.0, 'total_profit': 0.3},
                {'val_start': pd.Timestamp('2020-07-01'), 'val_end': pd.Timestamp('2020-12-31'),
                 'roi_percent': 5.0, 'num_bets': 12, 'wins': 7, 'total_wagered': 12.0, 'total_profit': 0.6},
                {'val_start': pd.Timestamp('2021-01-01'), 'val_end': pd.Timestamp('2021-06-30'),
                 'roi_percent': 7.0, 'num_bets': 8, 'wins': 5, 'total_wagered': 8.0, 'total_profit': 0.56},
            ]
        }
        all_results = [
            {'generation': 1, 'best_fitness': 3.0, 'mean_roi': 5.0, 'std_roi': 2.0, 'cv': 0.4, 'avg_fitness': 2.5},
            {'generation': 2, 'best_fitness': 3.5, 'mean_roi': 5.5, 'std_roi': 1.8, 'cv': 0.33, 'avg_fitness': 3.0},
        ]
        config = {'split_months': 6, 'lambda_penalty': 1.0}
        
        save_results(
            output_dir=self.temp_dir,
            split_months=6,
            best_params=best_params,
            best_fitness=3.5,
            extended_result=extended_result,
            all_results=all_results,
            config=config
        )
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, '6m_best_params.json')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, '6m_per_split_roi.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, '6m_evolution.csv')))
    
    def test_json_contains_required_fields(self):
        """Test that JSON output contains all required fields"""
        import json
        
        best_params = {'k': 32, 'w_ko': 1.4, 'w_sub': 1.3,
                      'w_udec': 1.0, 'w_sdec': 0.7, 'w_mdec': 0.9}
        extended_result = {
            'mean_roi': 5.0, 'std_roi': 2.0, 'cv': 0.4,
            'per_split_roi': [5.0],
            'per_split_details': [
                {'val_start': pd.Timestamp('2020-01-01'), 'val_end': pd.Timestamp('2020-06-30'),
                 'roi_percent': 5.0, 'num_bets': 10, 'wins': 6, 'total_wagered': 10.0, 'total_profit': 0.5}
            ]
        }
        
        save_results(
            output_dir=self.temp_dir,
            split_months=6,
            best_params=best_params,
            best_fitness=3.0,
            extended_result=extended_result,
            all_results=None,
            config={'split_months': 6}
        )
        
        with open(os.path.join(self.temp_dir, '6m_best_params.json')) as f:
            data = json.load(f)
        
        self.assertIn('best_params', data)
        self.assertIn('best_fitness', data)
        self.assertIn('mean_roi', data)
        self.assertIn('std_roi', data)
        self.assertIn('config', data)


class TestCLIArguments(unittest.TestCase):
    """Tests for CLI argument parsing"""
    
    def test_argparse_required_arguments(self):
        """Test that required arguments are properly defined"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-file", type=str, required=True)
        parser.add_argument("--split-months", type=str, required=True)
        parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_penalty")
        parser.add_argument("--objective", choices=["mean_std", "cv"], default="mean_std")
        parser.add_argument("--window-type", choices=["expanding", "rolling"], default="expanding")
        parser.add_argument("--generations", type=int, default=30)
        parser.add_argument("--population", type=int, default=30)
        parser.add_argument("--random-seed", type=int, default=None)
        
        # Test parsing with required args
        args = parser.parse_args([
            "--data-file", "data.csv",
            "--split-months", "6"
        ])
        
        self.assertEqual(args.data_file, "data.csv")
        self.assertEqual(args.split_months, "6")
        self.assertEqual(args.lambda_penalty, 1.0)
        self.assertEqual(args.objective, "mean_std")
    
    def test_argparse_split_months_list(self):
        """Test parsing comma-separated split months"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--split-months", type=str, required=True)
        
        args = parser.parse_args(["--split-months", "2,6,12"])
        
        split_months_list = [int(x.strip()) for x in args.split_months.split(',')]
        self.assertEqual(split_months_list, [2, 6, 12])
    
    def test_argparse_lambda_custom(self):
        """Test custom lambda penalty"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_penalty")
        
        args = parser.parse_args(["--lambda", "2.5"])
        self.assertEqual(args.lambda_penalty, 2.5)
    
    def test_argparse_cv_objective(self):
        """Test CV objective selection"""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--objective", choices=["mean_std", "cv"], default="mean_std")
        
        args = parser.parse_args(["--objective", "cv"])
        self.assertEqual(args.objective, "cv")


class TestFitnessCalculations(unittest.TestCase):
    """Tests for fitness calculation logic"""
    
    def test_higher_mean_roi_increases_fitness(self):
        """Test that higher mean ROI gives higher fitness (mean_std objective)"""
        # Mock results with different mean ROIs
        roi_values_low = [0.0, 1.0, 2.0]  # mean = 1.0
        roi_values_high = [5.0, 6.0, 7.0]  # mean = 6.0
        
        # Calculate fitness manually
        mean_low = np.mean(roi_values_low)
        std_low = np.std(roi_values_low, ddof=1)
        fitness_low = mean_low - 1.0 * std_low
        
        mean_high = np.mean(roi_values_high)
        std_high = np.std(roi_values_high, ddof=1)
        fitness_high = mean_high - 1.0 * std_high
        
        # Higher mean should give higher fitness (same std)
        self.assertGreater(fitness_high, fitness_low)
    
    def test_lower_std_roi_increases_fitness(self):
        """Test that lower std ROI gives higher fitness (mean_std objective)"""
        # Mock results with different std ROIs but same mean
        roi_values_consistent = [5.0, 5.0, 5.0]  # std = 0
        roi_values_variable = [0.0, 5.0, 10.0]  # std > 0
        
        mean1 = np.mean(roi_values_consistent)
        std1 = np.std(roi_values_consistent, ddof=1) if len(roi_values_consistent) > 1 else 0
        fitness_consistent = mean1 - 1.0 * std1
        
        mean2 = np.mean(roi_values_variable)
        std2 = np.std(roi_values_variable, ddof=1)
        fitness_variable = mean2 - 1.0 * std2
        
        # Lower variance should give higher fitness
        self.assertGreater(fitness_consistent, fitness_variable)
    
    def test_cv_calculation(self):
        """Test coefficient of variation calculation"""
        roi_values = [10.0, 12.0, 8.0]
        mean_roi = np.mean(roi_values)
        std_roi = np.std(roi_values, ddof=1)
        cv = std_roi / abs(mean_roi)
        
        # CV should be reasonable
        self.assertGreater(cv, 0)
        self.assertLess(cv, 1)  # For these values
    
    def test_cv_handles_zero_mean(self):
        """Test that CV handles zero or near-zero mean correctly"""
        roi_values = [-1.0, 0.0, 1.0]  # Mean = 0
        mean_roi = np.mean(roi_values)
        std_roi = np.std(roi_values, ddof=1)
        
        # When mean is near zero, CV should be inf or very high
        if abs(mean_roi) > 0.001:
            cv = std_roi / abs(mean_roi)
        else:
            cv = float('inf') if std_roi > 0 else 0.0
        
        # CV should handle this gracefully
        self.assertTrue(cv == float('inf') or cv >= 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
