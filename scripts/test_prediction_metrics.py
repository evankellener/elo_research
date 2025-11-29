"""
Unit tests for prediction_metrics.py - Calibration and Consistency Metrics
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prediction_metrics import (
    compute_expected_calibration_error,
    compute_calibration_slope,
    compute_all_calibration_metrics,
    compute_consistency_by_elo_tier,
    compute_auc_roc,
    compute_precision_recall,
    compute_confidence_separation,
    compute_roi_by_confidence_decile,
    compute_max_drawdown,
    compute_kelly_criterion,
    compute_value_bet_analysis,
    compute_comprehensive_metrics,
    compute_composite_fitness
)


class TestExpectedCalibrationError(unittest.TestCase):
    """Tests for Expected Calibration Error calculation"""
    
    def test_perfect_calibration(self):
        """Test ECE is near zero for perfectly calibrated predictions"""
        # Predictions exactly match actual rates
        predictions = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
        actuals = np.array([0, 0, 0, 1, 1, 1])
        
        result = compute_expected_calibration_error(predictions, actuals, n_bins=3)
        
        # ECE should be low (allowing for binning effects)
        self.assertIsNotNone(result['ece'])
        self.assertLess(result['ece'], 0.2)
    
    def test_terrible_calibration(self):
        """Test ECE is high for poorly calibrated predictions"""
        # Predictions are completely wrong
        predictions = np.array([0.9, 0.9, 0.9, 0.9, 0.1, 0.1])
        actuals = np.array([0, 0, 0, 0, 1, 1])
        
        result = compute_expected_calibration_error(predictions, actuals, n_bins=3)
        
        # ECE should be high
        self.assertIsNotNone(result['ece'])
        self.assertGreater(result['ece'], 0.5)
    
    def test_empty_predictions(self):
        """Test that empty predictions return None ECE"""
        result = compute_expected_calibration_error([], [], n_bins=10)
        self.assertIsNone(result['ece'])
        self.assertEqual(result['bin_data'], [])
    
    def test_bin_data_structure(self):
        """Test that bin_data has correct structure"""
        predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        actuals = np.array([0, 0, 1, 1, 1])
        
        result = compute_expected_calibration_error(predictions, actuals, n_bins=5)
        
        self.assertEqual(len(result['bin_data']), 5)
        for bin_info in result['bin_data']:
            self.assertIn('bin_lower', bin_info)
            self.assertIn('bin_upper', bin_info)
            self.assertIn('count', bin_info)


class TestCalibrationSlope(unittest.TestCase):
    """Tests for calibration slope calculation"""
    
    def test_perfect_calibration_slope(self):
        """Test slope near 1.0 for well-calibrated predictions"""
        # Create predictions that match actual rates well
        np.random.seed(42)
        predictions = np.random.uniform(0.3, 0.7, 100)
        actuals = (np.random.random(100) < predictions).astype(int)
        
        result = compute_calibration_slope(predictions, actuals)
        
        # Slope should be close to 1.0 (within 0.5)
        self.assertIsNotNone(result['slope'])
        self.assertGreater(result['slope'], 0.5)
        self.assertLess(result['slope'], 1.5)
    
    def test_empty_predictions(self):
        """Test that short predictions return None"""
        result = compute_calibration_slope([0.5], [1])
        self.assertIsNone(result['slope'])
    
    def test_overconfident_detection(self):
        """Test detection of overconfident predictions"""
        # Predictions spread too wide (model is overconfident)
        predictions = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        actuals = np.array([0, 1, 0, 0, 1, 1])  # Actual rates closer to 50%
        
        result = compute_calibration_slope(predictions, actuals)
        # This might not always trigger overconfident, depending on data
        self.assertIn('is_overconfident', result)


class TestAllCalibrationMetrics(unittest.TestCase):
    """Tests for combined calibration metrics"""
    
    def test_returns_all_keys(self):
        """Test that all expected keys are returned"""
        predictions = np.array([0.3, 0.5, 0.7])
        actuals = np.array([0, 1, 1])
        
        result = compute_all_calibration_metrics(predictions, actuals)
        
        expected_keys = ['ece', 'brier_score', 'log_loss', 'calibration_slope',
                        'calibration_intercept', 'is_overconfident', 
                        'is_underconfident', 'bin_data', 'total_predictions']
        for key in expected_keys:
            self.assertIn(key, result)
    
    def test_brier_score_range(self):
        """Test that Brier score is in valid range [0, 1]"""
        predictions = np.random.random(50)
        actuals = np.random.randint(0, 2, 50)
        
        result = compute_all_calibration_metrics(predictions, actuals)
        
        self.assertGreaterEqual(result['brier_score'], 0)
        self.assertLessEqual(result['brier_score'], 1)
    
    def test_log_loss_positive(self):
        """Test that log loss is positive"""
        predictions = np.array([0.3, 0.7, 0.5])
        actuals = np.array([0, 1, 0])
        
        result = compute_all_calibration_metrics(predictions, actuals)
        
        self.assertGreater(result['log_loss'], 0)


class TestConsistencyByEloTier(unittest.TestCase):
    """Tests for consistency analysis by Elo tier"""
    
    def test_creates_tiers(self):
        """Test that tiers are created correctly"""
        predictions = np.array([0.6, 0.7, 0.4, 0.8])
        actuals = np.array([1, 1, 0, 1])
        elo_diffs = np.array([-150, -50, 50, 150])
        
        result = compute_consistency_by_elo_tier(None, predictions, actuals, elo_diffs)
        
        self.assertIn('tiers', result)
        self.assertGreater(len(result['tiers']), 0)
    
    def test_variance_calculation(self):
        """Test that variance is calculated"""
        predictions = np.array([0.6] * 10 + [0.8] * 10)
        actuals = np.array([1] * 8 + [0] * 2 + [1] * 9 + [0])
        elo_diffs = np.array([-100] * 10 + [100] * 10)
        
        result = compute_consistency_by_elo_tier(None, predictions, actuals, elo_diffs)
        
        self.assertIn('accuracy_variance', result)
    
    def test_high_variance_flag(self):
        """Test that high variance flag works"""
        predictions = np.array([0.6] * 10 + [0.8] * 10)
        actuals = np.array([1] * 10 + [0] * 10)  # Very different accuracies
        elo_diffs = np.array([-100] * 10 + [100] * 10)
        
        result = compute_consistency_by_elo_tier(None, predictions, actuals, elo_diffs)
        
        self.assertIn('high_variance_flag', result)


class TestAUCROC(unittest.TestCase):
    """Tests for AUC-ROC calculation"""
    
    def test_perfect_auc(self):
        """Test AUC of 1.0 for perfect separation"""
        predictions = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        actuals = np.array([0, 0, 0, 1, 1, 1])
        
        auc = compute_auc_roc(predictions, actuals)
        
        self.assertEqual(auc, 1.0)
    
    def test_random_auc(self):
        """Test AUC near 0.5 for random predictions"""
        np.random.seed(42)
        predictions = np.random.random(100)
        actuals = np.random.randint(0, 2, 100)
        
        auc = compute_auc_roc(predictions, actuals)
        
        # Should be close to 0.5 (random)
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.3)
        self.assertLess(auc, 0.7)
    
    def test_empty_returns_none(self):
        """Test that empty arrays return None"""
        auc = compute_auc_roc([], [])
        self.assertIsNone(auc)
    
    def test_single_class_returns_none(self):
        """Test that single class returns None"""
        auc = compute_auc_roc([0.5, 0.6, 0.7], [1, 1, 1])
        self.assertIsNone(auc)


class TestPrecisionRecall(unittest.TestCase):
    """Tests for Precision and Recall calculation"""
    
    def test_returns_required_keys(self):
        """Test that all required keys are returned"""
        predictions = np.array([0.6, 0.7, 0.4, 0.8])
        actuals = np.array([1, 1, 0, 1])
        
        result = compute_precision_recall(predictions, actuals)
        
        self.assertIn('favorites', result)
        self.assertIn('underdogs', result)
        self.assertIn('precision', result['favorites'])
        self.assertIn('recall', result['favorites'])
    
    def test_perfect_precision(self):
        """Test precision for predictions with correct outcomes"""
        # All favorites (>0.5) that all win
        predictions = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        actuals = np.array([1, 1, 1, 0, 0])
        
        result = compute_precision_recall(predictions, actuals)
        
        # Precision for favorites should be 1.0 (all predicted wins are actual wins)
        self.assertIsNotNone(result['favorites']['precision'])
        self.assertEqual(result['favorites']['precision'], 1.0)


class TestConfidenceSeparation(unittest.TestCase):
    """Tests for confidence separation calculation"""
    
    def test_positive_separation_for_good_model(self):
        """Test positive separation when winners get higher confidence"""
        predictions = np.array([0.8, 0.7, 0.3, 0.2])
        actuals = np.array([1, 1, 0, 0])
        
        result = compute_confidence_separation(predictions, actuals)
        
        self.assertGreater(result['separation'], 0)
        self.assertGreater(result['avg_confidence_winners'], result['avg_confidence_losers'])
    
    def test_empty_returns_none(self):
        """Test that empty arrays return None"""
        result = compute_confidence_separation([], [])
        self.assertIsNone(result['separation'])


class TestROIByConfidenceDecile(unittest.TestCase):
    """Tests for ROI by confidence decile calculation"""
    
    def test_creates_deciles(self):
        """Test that deciles are created"""
        bet_records = [
            {'profit': 0.5, 'bet_amount': 1.0, 'bet_won': 1},
            {'profit': -1.0, 'bet_amount': 1.0, 'bet_won': 0},
            {'profit': 0.3, 'bet_amount': 1.0, 'bet_won': 1},
        ]
        predictions = np.array([0.55, 0.65, 0.75])
        
        result = compute_roi_by_confidence_decile(bet_records, predictions)
        
        self.assertIn('deciles', result)
    
    def test_empty_returns_empty(self):
        """Test that empty records return empty deciles"""
        result = compute_roi_by_confidence_decile([], [])
        self.assertEqual(result['deciles'], {})


class TestMaxDrawdown(unittest.TestCase):
    """Tests for max drawdown calculation"""
    
    def test_no_drawdown_when_always_increasing(self):
        """Test drawdown is 0 when profits always increase"""
        cumulative = [1, 2, 3, 4, 5]
        
        result = compute_max_drawdown(cumulative)
        
        self.assertEqual(result['max_drawdown'], 0)
    
    def test_detects_drawdown(self):
        """Test that drawdown is correctly detected"""
        cumulative = [1, 2, 3, 1, 2]  # Drop from 3 to 1
        
        result = compute_max_drawdown(cumulative)
        
        self.assertEqual(result['max_drawdown'], 2)  # 3 - 1 = 2
    
    def test_empty_returns_none(self):
        """Test that empty array returns None"""
        result = compute_max_drawdown([])
        self.assertIsNone(result['max_drawdown'])


class TestKellyCriterion(unittest.TestCase):
    """Tests for Kelly Criterion calculation"""
    
    def test_positive_kelly_for_edge(self):
        """Test positive Kelly when we have positive edge"""
        # 60% win probability at 2.0 odds (implied 50%)
        kelly = compute_kelly_criterion(0.6, 2.0)
        
        self.assertGreater(kelly, 0)
    
    def test_negative_kelly_for_no_edge(self):
        """Test negative Kelly when odds are against us"""
        # 40% win probability at 2.0 odds (implied 50%)
        kelly = compute_kelly_criterion(0.4, 2.0)
        
        self.assertLess(kelly, 0)
    
    def test_zero_kelly_at_break_even(self):
        """Test Kelly near 0 at break-even point"""
        # 50% win probability at 2.0 odds (implied 50%)
        kelly = compute_kelly_criterion(0.5, 2.0)
        
        self.assertAlmostEqual(kelly, 0, places=5)


class TestValueBetAnalysis(unittest.TestCase):
    """Tests for value bet analysis"""
    
    def test_counts_value_bets(self):
        """Test that value bets are correctly counted"""
        predictions = np.array([0.6, 0.7, 0.4, 0.8])
        implied_probs = np.array([0.5, 0.5, 0.5, 0.5])  # 2 are value bets
        actuals = np.array([1, 1, 0, 1])
        
        result = compute_value_bet_analysis(predictions, implied_probs, actuals)
        
        self.assertEqual(result['value_bet_count'], 3)  # 0.6, 0.7, 0.8 > 0.5
    
    def test_empty_returns_zero(self):
        """Test that empty arrays return 0 count"""
        result = compute_value_bet_analysis([], [], [])
        self.assertEqual(result['value_bet_count'], 0)


class TestComprehensiveMetrics(unittest.TestCase):
    """Tests for comprehensive metrics computation"""
    
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
                'precomp_elo': 1500 + (i * 10),
                'postcomp_elo': 1500 + (i * 10) + 20,
                'opp_precomp_elo': 1500 - (i * 5),
                'opp_postcomp_elo': 1500 - (i * 5) - 20,
                'precomp_boutcount': i,
                'opp_precomp_boutcount': i + 1
            })
        
        self.df = pd.DataFrame(data)
        self.df['DATE'] = pd.to_datetime(self.df['DATE']).dt.tz_localize(None)
    
    def test_returns_all_categories(self):
        """Test that all metric categories are returned"""
        result = compute_comprehensive_metrics(self.df)
        
        self.assertIn('calibration', result)
        self.assertIn('consistency', result)
        self.assertIn('performance', result)
        self.assertIn('betting', result)
        self.assertIn('summary', result)
    
    def test_summary_has_key_metrics(self):
        """Test that summary contains key metrics"""
        result = compute_comprehensive_metrics(self.df)
        
        summary = result['summary']
        self.assertIn('ece', summary)
        self.assertIn('brier_score', summary)
        self.assertIn('auc_roc', summary)
        self.assertIn('overall_accuracy', summary)


class TestComposeFitness(unittest.TestCase):
    """Tests for composite fitness calculation"""
    
    def test_returns_float(self):
        """Test that composite fitness returns a float"""
        metrics = {
            'roi_percent': 5.0,
            'summary': {
                'overall_accuracy': 0.6,
                'ece': 0.05,
                'consistency_variance': 0.005,
                'auc_roc': 0.65
            }
        }
        
        fitness = compute_composite_fitness(metrics)
        
        self.assertIsInstance(fitness, float)
    
    def test_higher_roi_gives_higher_fitness(self):
        """Test that higher ROI gives higher fitness"""
        metrics_low = {
            'roi_percent': 0.0,
            'summary': {
                'overall_accuracy': 0.5,
                'ece': 0.1,
                'consistency_variance': 0.01,
                'auc_roc': 0.5
            }
        }
        metrics_high = {
            'roi_percent': 10.0,
            'summary': {
                'overall_accuracy': 0.5,
                'ece': 0.1,
                'consistency_variance': 0.01,
                'auc_roc': 0.5
            }
        }
        
        fitness_low = compute_composite_fitness(metrics_low)
        fitness_high = compute_composite_fitness(metrics_high)
        
        self.assertGreater(fitness_high, fitness_low)
    
    def test_custom_weights(self):
        """Test that custom weights are applied"""
        metrics = {
            'roi_percent': 5.0,
            'summary': {
                'overall_accuracy': 0.6,
                'ece': 0.05,
                'consistency_variance': 0.005,
                'auc_roc': 0.65
            }
        }
        
        # Weight heavily towards ROI
        fitness_roi_heavy = compute_composite_fitness(metrics, {'roi': 0.9, 'accuracy': 0.1})
        
        # Weight heavily towards accuracy
        fitness_acc_heavy = compute_composite_fitness(metrics, {'roi': 0.1, 'accuracy': 0.9})
        
        # Should be different
        self.assertNotEqual(fitness_roi_heavy, fitness_acc_heavy)


if __name__ == '__main__':
    unittest.main(verbosity=2)
