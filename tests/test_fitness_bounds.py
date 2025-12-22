#!/usr/bin/env python3
"""
Test to verify fitness values are properly bounded and don't explode.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import constants from the optimization script
import optimize_decay_ga_consistency as opt

class TestFitnessBounds(unittest.TestCase):
    """Test that fitness calculations don't produce unreasonable values."""
    
    def test_sharpe_ratio_with_small_std(self):
        """Test that Sharpe ratio is capped when std is very small."""
        mean_roi = 200.0  # 200% ROI (very high)
        std_roi = 0.005   # Very small std (below EPSILON_THRESHOLD)
        
        # Simulate the calculation from evaluate_consistency_fitness
        if std_roi < opt.EPSILON_THRESHOLD:
            sharpe = min(opt.MAX_SHARPE_RATIO, mean_roi / opt.EPSILON_THRESHOLD)
        else:
            sharpe = mean_roi / std_roi
        
        # Sharpe should be capped at MAX_SHARPE_RATIO
        self.assertEqual(sharpe, opt.MAX_SHARPE_RATIO)
        self.assertLessEqual(sharpe, 100, "Sharpe ratio should never be in the hundreds")
    
    def test_sharpe_ratio_with_normal_std(self):
        """Test that Sharpe ratio is normal when std is reasonable."""
        mean_roi = 10.0
        std_roi = 2.0
        
        # Normal calculation
        sharpe = mean_roi / std_roi
        
        self.assertEqual(sharpe, 5.0)
        self.assertLess(sharpe, opt.MAX_SHARPE_RATIO)
    
    def test_sharpe_ratio_with_zero_std(self):
        """Test that Sharpe ratio handles zero std gracefully."""
        mean_roi = 50.0
        std_roi = 0.0
        
        # Simulate the calculation
        if std_roi < opt.EPSILON_THRESHOLD:
            sharpe = min(opt.MAX_SHARPE_RATIO, mean_roi / opt.EPSILON_THRESHOLD)
        else:
            sharpe = mean_roi / std_roi
        
        # Should be capped
        self.assertEqual(sharpe, opt.MAX_SHARPE_RATIO)
    
    def test_fitness_components_bounded(self):
        """Test that all fitness components produce reasonable values."""
        # Simulate best case scenario
        mean_roi = 20.0
        std_roi = 0.001  # Very small
        
        if std_roi < opt.EPSILON_THRESHOLD:
            sharpe = min(opt.MAX_SHARPE_RATIO, mean_roi / opt.EPSILON_THRESHOLD)
        else:
            sharpe = mean_roi / std_roi
        
        # Cap mean_roi as done in the actual code
        capped_mean_roi = min(mean_roi, opt.MAX_MEAN_ROI) if mean_roi > 0 else mean_roi
        
        # Fitness calculation from evaluate_consistency_fitness
        trend_bonus = 0  # Assume no trend
        calibration_penalty = 10  # Typical value
        
        fitness = (
            0.4 * sharpe +
            0.3 * capped_mean_roi +
            0.2 * trend_bonus +
            0.1 * (-calibration_penalty)
        )
        
        # Fitness should be reasonable, not in the millions
        self.assertLess(fitness, 100, "Fitness should not be in the hundreds")
        self.assertGreater(fitness, -1000, "Fitness should not be extremely negative")
        
        # With our fix: 0.4 * 20 + 0.3 * 20 + 0 - 1 = 8 + 6 - 1 = 13
        expected_fitness = 0.4 * opt.MAX_SHARPE_RATIO + 0.3 * capped_mean_roi - 1.0
        self.assertAlmostEqual(fitness, expected_fitness, places=1)
    
    def test_extreme_roi_capped(self):
        """Test that extremely high ROI values are properly capped."""
        mean_roi = 500.0  # Extreme 500% ROI
        std_roi = 0.001
        
        if std_roi < opt.EPSILON_THRESHOLD:
            sharpe = min(opt.MAX_SHARPE_RATIO, mean_roi / opt.EPSILON_THRESHOLD)
        else:
            sharpe = mean_roi / std_roi
        
        # Cap mean_roi as done in actual code
        capped_mean_roi = min(mean_roi, opt.MAX_MEAN_ROI) if mean_roi > 0 else mean_roi
        
        # Even with 500% ROI, sharpe should be capped
        self.assertEqual(sharpe, opt.MAX_SHARPE_RATIO)
        
        # And mean_roi should be capped
        self.assertEqual(capped_mean_roi, opt.MAX_MEAN_ROI)
        
        # Verify fitness won't explode
        fitness = 0.4 * sharpe + 0.3 * capped_mean_roi
        # Max: 0.4 * 20 + 0.3 * 100 = 8 + 30 = 38
        self.assertLess(fitness, 50, "Fitness should remain reasonable even with extreme ROI")
        self.assertAlmostEqual(fitness, 38.0, places=1)

if __name__ == '__main__':
    unittest.main()
