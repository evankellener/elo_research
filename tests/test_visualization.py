"""
Test for visualization functions, specifically plot_ga_optimization_history
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
import tempfile
from elo.visualization import plot_ga_optimization_history


class TestGAVisualization(unittest.TestCase):
    """Test cases for GA optimization visualization."""
    
    def setUp(self):
        """Create sample history dataframe for testing."""
        self.history_df = pd.DataFrame({
            'generation': [1, 2, 3, 4, 5],
            'best_fitness': [0.5, 0.55, 0.58, 0.58, 0.60],
            'avg_fitness': [0.45, 0.50, 0.52, 0.54, 0.56],
            'worst_fitness': [0.35, 0.40, 0.42, 0.45, 0.48],
            'best_genes': [
                {'k': 100, 'denominator': 400},
                {'k': 120, 'denominator': 380},
                {'k': 110, 'denominator': 390},
                {'k': 105, 'denominator': 395},
                {'k': 108, 'denominator': 392}
            ]
        })
    
    def test_plot_generation_accuracy_mode(self):
        """Test that plot is generated for accuracy optimization mode."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = plot_ga_optimization_history(
                self.history_df,
                optimize_for='accuracy',
                save_path=tmp_path,
                show_plot=False
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_plot_generation_roi_mode(self):
        """Test that plot is generated for ROI optimization mode."""
        # Modify fitness values to be in ROI range
        roi_df = self.history_df.copy()
        roi_df['best_fitness'] = [-0.05, -0.03, 0.01, 0.05, 0.08]
        roi_df['avg_fitness'] = [-0.10, -0.08, -0.04, 0.00, 0.03]
        roi_df['worst_fitness'] = [-0.20, -0.15, -0.10, -0.05, -0.02]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = plot_ga_optimization_history(
                roi_df,
                optimize_for='roi',
                save_path=tmp_path,
                show_plot=False
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_plot_generation_log_loss_mode(self):
        """Test that plot is generated for log_loss optimization mode."""
        # Modify fitness values to be in log_loss fitness range (exp(-log_loss))
        log_loss_df = self.history_df.copy()
        log_loss_df['best_fitness'] = [0.3, 0.4, 0.45, 0.48, 0.50]
        log_loss_df['avg_fitness'] = [0.25, 0.30, 0.35, 0.38, 0.40]
        log_loss_df['worst_fitness'] = [0.15, 0.20, 0.25, 0.28, 0.30]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = plot_ga_optimization_history(
                log_loss_df,
                optimize_for='log_loss',
                save_path=tmp_path,
                show_plot=False
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_plot_generation_composite_mode(self):
        """Test that plot is generated for composite optimization mode."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            fig = plot_ga_optimization_history(
                self.history_df,
                optimize_for='composite',
                save_path=tmp_path,
                show_plot=False
            )
            
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_empty_dataframe(self):
        """Test that empty dataframe is handled gracefully."""
        empty_df = pd.DataFrame()
        fig = plot_ga_optimization_history(
            empty_df,
            optimize_for='accuracy',
            save_path=None,
            show_plot=False
        )
        self.assertIsNone(fig)
    
    def test_plot_without_save(self):
        """Test that plot can be generated without saving to file."""
        fig = plot_ga_optimization_history(
            self.history_df,
            optimize_for='accuracy',
            save_path=None,
            show_plot=False
        )
        self.assertIsNotNone(fig)
    
    def test_invalid_genes_handling(self):
        """Test that invalid genes in best_genes are handled gracefully."""
        invalid_df = pd.DataFrame({
            'generation': [1, 2, 3],
            'best_fitness': [0.5, 0.55, 0.58],
            'avg_fitness': [0.45, 0.50, 0.52],
            'worst_fitness': [0.35, 0.40, 0.42],
            'best_genes': [
                {'k': 100, 'denominator': 400},
                None,  # Invalid: None instead of dict
                'invalid'  # Invalid: string instead of dict
            ]
        })
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Should not raise an exception
            fig = plot_ga_optimization_history(
                invalid_df,
                optimize_for='accuracy',
                save_path=tmp_path,
                show_plot=False
            )
            self.assertIsNotNone(fig)
            self.assertTrue(os.path.exists(tmp_path))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()
