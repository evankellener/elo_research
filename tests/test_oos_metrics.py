"""
Test OOS (Out-of-Sample) metrics functionality in example_ga_optimization.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from examples.example_ga_optimization import test_out_of_sample_metrics
from optimization.optimal_k_with_mov import run_basic_elo


class TestOOSMetrics(unittest.TestCase):
    """Test out-of-sample metrics calculation."""
    
    def setUp(self):
        """Create synthetic test data."""
        # Create synthetic training data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        self.train_df = pd.DataFrame({
            'DATE': dates,
            'FIGHTER': ['Fighter_A', 'Fighter_B', 'Fighter_A', 'Fighter_C', 'Fighter_B',
                       'Fighter_D', 'Fighter_A', 'Fighter_C', 'Fighter_B', 'Fighter_D'],
            'opp_FIGHTER': ['Fighter_B', 'Fighter_A', 'Fighter_C', 'Fighter_A', 'Fighter_D',
                           'Fighter_B', 'Fighter_D', 'Fighter_B', 'Fighter_C', 'Fighter_A'],
            'result': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        
        # Create synthetic test data (OOS)
        test_dates = pd.date_range('2023-01-11', periods=5, freq='D')
        self.test_df = pd.DataFrame({
            'date': test_dates,
            'fighter': ['Fighter_A', 'Fighter_B', 'Fighter_C', 'Fighter_D', 'Fighter_A'],
            'opp_fighter': ['Fighter_B', 'Fighter_C', 'Fighter_D', 'Fighter_A', 'Fighter_C'],
            'result': [1, 0, 1, 0, 1]
        })
    
    def test_oos_metrics_basic(self):
        """Test that OOS metrics can be calculated."""
        # Train Elo on training data
        df_with_elo = run_basic_elo(
            self.train_df.copy(),
            k=32,
            base_elo=1500,
            denominator=400,
            use_mov=False
        )
        
        # Calculate OOS metrics
        metrics = test_out_of_sample_metrics(
            df_with_elo,
            self.test_df,
            denominator=400,
            base_elo=1500,
            confidence_threshold=50,
            optimize_for="accuracy",
            verbose=False
        )
        
        # Verify metrics structure
        self.assertIn('accuracy', metrics)
        self.assertIn('roi', metrics)
        self.assertIn('log_loss', metrics)
        self.assertIn('n_predictions', metrics)
        self.assertIn('n_bets', metrics)
        
        # Verify metrics are in valid ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreater(metrics['n_predictions'], 0)
        self.assertIsInstance(metrics['log_loss'], float)
        self.assertIsInstance(metrics['roi'], float)
    
    def test_oos_metrics_with_different_denominators(self):
        """Test OOS metrics with different denominator values."""
        df_with_elo = run_basic_elo(
            self.train_df.copy(),
            k=32,
            base_elo=1500,
            denominator=400,
            use_mov=False
        )
        
        # Test with standard denominator
        metrics_400 = test_out_of_sample_metrics(
            df_with_elo,
            self.test_df,
            denominator=400,
            base_elo=1500,
            verbose=False
        )
        
        # Test with different denominator
        metrics_600 = test_out_of_sample_metrics(
            df_with_elo,
            self.test_df,
            denominator=600,
            base_elo=1500,
            verbose=False
        )
        
        # Both should produce valid metrics
        self.assertGreater(metrics_400['n_predictions'], 0)
        self.assertGreater(metrics_600['n_predictions'], 0)
        
        # Metrics might differ due to different probability calculations
        # but both should be in valid ranges
        self.assertGreaterEqual(metrics_400['accuracy'], 0.0)
        self.assertLessEqual(metrics_400['accuracy'], 1.0)
        self.assertGreaterEqual(metrics_600['accuracy'], 0.0)
        self.assertLessEqual(metrics_600['accuracy'], 1.0)
    
    def test_oos_with_real_data(self):
        """Test OOS metrics with real data if available."""
        data_path = "data/interleaved_cleaned.csv"
        test_path = "data/past3_events.csv"
        
        if not os.path.exists(data_path) or not os.path.exists(test_path):
            self.skipTest("Real data files not found")
        
        # Load real data
        df = pd.read_csv(data_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)
        
        # Preprocess
        df["result"] = pd.to_numeric(df["result"], errors="coerce")
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").reset_index(drop=True)
        
        # Use subset for faster test
        df = df.tail(1000).copy()
        
        # Train Elo
        df_with_elo = run_basic_elo(
            df.copy(),
            k=32,
            base_elo=1500,
            denominator=400,
            use_mov=False
        )
        
        # Calculate OOS metrics
        metrics = test_out_of_sample_metrics(
            df_with_elo,
            test_df,
            denominator=400,
            base_elo=1500,
            verbose=False
        )
        
        # Verify we got valid predictions
        self.assertGreater(metrics['n_predictions'], 0, 
                          "Should have at least one OOS prediction")
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        print(f"\nReal data OOS test:")
        print(f"  Predictions: {metrics['n_predictions']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  ROI: {metrics['roi']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")


if __name__ == '__main__':
    unittest.main()
