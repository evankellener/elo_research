"""
Unit tests for time_splitter.py module.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from time_splitter import (
    detect_time_column,
    create_time_splits,
    create_time_splits_with_offset,
    get_split_info,
    _parse_offset_months,
)


class TestDetectTimeColumn(unittest.TestCase):
    """Tests for detect_time_column function"""
    
    def test_detects_DATE(self):
        """Test detection of DATE column"""
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        self.assertEqual(detect_time_column(df), 'DATE')
    
    def test_detects_date_lowercase(self):
        """Test detection of lowercase date column"""
        df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        self.assertEqual(detect_time_column(df), 'date')
    
    def test_detects_timestamp(self):
        """Test detection of timestamp column"""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        self.assertEqual(detect_time_column(df), 'timestamp')
    
    def test_returns_none_when_not_found(self):
        """Test that None is returned when no time column found"""
        df = pd.DataFrame({
            'created_at': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        self.assertIsNone(detect_time_column(df))
    
    def test_prefers_DATE_over_date(self):
        """Test that DATE is preferred over lowercase date"""
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=5),
            'date': pd.date_range('2020-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        self.assertEqual(detect_time_column(df), 'DATE')


class TestCreateTimeSplits(unittest.TestCase):
    """Tests for create_time_splits function"""
    
    def setUp(self):
        """Create sample data for testing"""
        # Create 3 years of monthly data (36 months)
        dates = pd.date_range('2020-01-01', periods=36, freq='ME')
        self.df = pd.DataFrame({
            'DATE': dates,
            'value': range(36),
            'FIGHTER': ['A'] * 36,
            'opp_FIGHTER': ['B'] * 36,
            'result': [1, 0] * 18,
        })
    
    def test_creates_splits_with_default_params(self):
        """Test that splits are created with default parameters"""
        splits = create_time_splits(self.df, months=6)
        
        # Should create multiple splits
        self.assertGreater(len(splits), 0)
        
        # Each split should have train and val DataFrames
        for train_df, val_df in splits:
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertIsInstance(val_df, pd.DataFrame)
            self.assertGreater(len(train_df), 0)
            self.assertGreater(len(val_df), 0)
    
    def test_expanding_window_grows(self):
        """Test that expanding window training data grows with each split"""
        splits = create_time_splits(self.df, months=6, window_type="expanding")
        
        if len(splits) > 1:
            # Training data should grow
            train_sizes = [len(train) for train, _ in splits]
            for i in range(1, len(train_sizes)):
                self.assertGreater(train_sizes[i], train_sizes[i-1])
    
    def test_rolling_window_fixed_size(self):
        """Test that rolling window training data is approximately fixed size"""
        splits = create_time_splits(self.df, months=6, window_type="rolling")
        
        if len(splits) > 1:
            # Training data should be approximately the same size
            train_sizes = [len(train) for train, _ in splits]
            # Allow some variance due to month length differences
            size_variance = max(train_sizes) - min(train_sizes)
            avg_size = sum(train_sizes) / len(train_sizes)
            self.assertLess(size_variance / avg_size, 0.5)  # Less than 50% variance
    
    def test_validation_periods_non_overlapping(self):
        """Test that validation periods don't overlap"""
        splits = create_time_splits(self.df, months=6)
        
        for i in range(len(splits) - 1):
            _, val1 = splits[i]
            _, val2 = splits[i + 1]
            
            val1_end = val1['DATE'].max()
            val2_start = val2['DATE'].min()
            
            self.assertLess(val1_end, val2_start)
    
    def test_auto_detects_time_column(self):
        """Test auto-detection of time column"""
        splits = create_time_splits(self.df, months=6, time_column=None)
        self.assertGreater(len(splits), 0)
    
    def test_raises_error_for_missing_time_column(self):
        """Test that error is raised when time column not found"""
        df = self.df.rename(columns={'DATE': 'created_at'})
        with self.assertRaises(ValueError):
            create_time_splits(df, months=6, time_column=None)
    
    def test_raises_error_for_invalid_window_type(self):
        """Test that error is raised for invalid window type"""
        with self.assertRaises(ValueError):
            create_time_splits(self.df, months=6, window_type="invalid")
    
    def test_handles_short_data(self):
        """Test handling of data too short for splits"""
        short_df = self.df.head(3)  # Only 3 months
        splits = create_time_splits(short_df, months=6, min_train_months=6)
        
        # May create no splits or very few
        self.assertIsInstance(splits, list)
    
    def test_min_train_months_enforced(self):
        """Test that minimum training months requirement is enforced"""
        splits = create_time_splits(self.df, months=3, min_train_months=12)
        
        for train_df, _ in splits:
            train_months = (train_df['DATE'].max() - train_df['DATE'].min()).days / 30
            # Training should have at least the minimum (with some tolerance)
            self.assertGreaterEqual(train_months, 10)  # Allow slight variance


class TestCreateTimeSplitsWithOffset(unittest.TestCase):
    """Tests for create_time_splits_with_offset function"""
    
    def setUp(self):
        """Create sample data"""
        dates = pd.date_range('2020-01-01', periods=36, freq='ME')
        self.df = pd.DataFrame({
            'DATE': dates,
            'value': range(36),
            'FIGHTER': ['A'] * 36,
            'opp_FIGHTER': ['B'] * 36,
            'result': [1, 0] * 18,
        })
    
    def test_accepts_string_offset(self):
        """Test that string offset like '6M' works"""
        splits = create_time_splits_with_offset(self.df, '6M')
        self.assertGreater(len(splits), 0)
    
    def test_accepts_year_offset(self):
        """Test that year offset like '1Y' works"""
        splits = create_time_splits_with_offset(self.df, '1Y')
        self.assertGreater(len(splits), 0)
    
    def test_same_as_create_time_splits(self):
        """Test that results are similar to create_time_splits with same params"""
        splits1 = create_time_splits(self.df, months=6)
        splits2 = create_time_splits_with_offset(self.df, '6M')
        
        # Should have approximately same number of splits
        self.assertAlmostEqual(len(splits1), len(splits2), delta=1)


class TestParseOffsetMonths(unittest.TestCase):
    """Tests for _parse_offset_months helper function"""
    
    def test_parses_months(self):
        """Test parsing month offsets"""
        self.assertEqual(_parse_offset_months('1M'), 1)
        self.assertEqual(_parse_offset_months('6M'), 6)
        self.assertEqual(_parse_offset_months('12M'), 12)
    
    def test_parses_years(self):
        """Test parsing year offsets"""
        self.assertEqual(_parse_offset_months('1Y'), 12)
        self.assertEqual(_parse_offset_months('2Y'), 24)
    
    def test_case_insensitive(self):
        """Test that parsing is case-insensitive"""
        self.assertEqual(_parse_offset_months('6m'), 6)
        self.assertEqual(_parse_offset_months('1y'), 12)
    
    def test_raises_on_invalid(self):
        """Test that invalid formats raise ValueError"""
        with self.assertRaises(ValueError):
            _parse_offset_months('6W')  # Weeks not supported
        with self.assertRaises(ValueError):
            _parse_offset_months('invalid')


class TestGetSplitInfo(unittest.TestCase):
    """Tests for get_split_info function"""
    
    def setUp(self):
        """Create sample data and splits"""
        dates = pd.date_range('2020-01-01', periods=36, freq='ME')
        self.df = pd.DataFrame({
            'DATE': dates,
            'value': range(36),
            'FIGHTER': ['A'] * 36,
            'opp_FIGHTER': ['B'] * 36,
            'result': [1, 0] * 18,
        })
        self.splits = create_time_splits(self.df, months=6)
    
    def test_returns_dataframe(self):
        """Test that get_split_info returns a DataFrame"""
        info = get_split_info(self.splits)
        self.assertIsInstance(info, pd.DataFrame)
    
    def test_has_required_columns(self):
        """Test that all required columns are present"""
        info = get_split_info(self.splits)
        required_cols = ['split_idx', 'train_start', 'train_end', 'train_rows',
                        'val_start', 'val_end', 'val_rows']
        for col in required_cols:
            self.assertIn(col, info.columns)
    
    def test_correct_number_of_rows(self):
        """Test that info has correct number of rows"""
        info = get_split_info(self.splits)
        self.assertEqual(len(info), len(self.splits))
    
    def test_row_counts_match(self):
        """Test that row counts in info match actual splits"""
        info = get_split_info(self.splits)
        for i, (train, val) in enumerate(self.splits):
            row = info[info['split_idx'] == i].iloc[0]
            self.assertEqual(row['train_rows'], len(train))
            self.assertEqual(row['val_rows'], len(val))


class TestIntegrationWithRealData(unittest.TestCase):
    """Integration tests using actual data files"""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level resources"""
        cls.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_file = os.path.join(cls.project_root, 'data', 'interleaved_cleaned.csv')
    
    @unittest.skipIf(
        not os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'interleaved_cleaned.csv'
        )),
        "interleaved_cleaned.csv not found"
    )
    def test_splits_on_real_data(self):
        """Test creating splits on real fight data"""
        df = pd.read_csv(self.data_file, low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Create splits
        splits = create_time_splits(df, months=6)
        
        # Should create multiple splits
        self.assertGreater(len(splits), 0)
        
        # Get info
        info = get_split_info(splits)
        
        # All splits should have data
        self.assertTrue((info['train_rows'] > 0).all())
        self.assertTrue((info['val_rows'] > 0).all())
    
    @unittest.skipIf(
        not os.path.exists(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'interleaved_cleaned.csv'
        )),
        "interleaved_cleaned.csv not found"
    )
    def test_different_split_durations(self):
        """Test different split durations on real data"""
        df = pd.read_csv(self.data_file, low_memory=False)
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        for months in [2, 6, 12]:
            splits = create_time_splits(df, months=months)
            
            # Should create splits (number depends on data duration)
            self.assertIsInstance(splits, list)
            
            if len(splits) > 0:
                # Verify validation periods are approximately correct length
                for _, val in splits:
                    val_days = (val['DATE'].max() - val['DATE'].min()).days
                    # Allow flexibility - validation period may span up to the month duration
                    # The actual days can vary due to month boundaries
                    expected_max_days = months * 31 + 31  # Allow for variable month lengths
                    self.assertLessEqual(val_days, expected_max_days)


if __name__ == '__main__':
    unittest.main(verbosity=2)
