"""
Unit tests for time split filtering functionality.

Tests exercise create_time_splits() and filter_and_sample_splits() using
a small synthetic dataset. Tests are designed to be fast and deterministic.
"""
import os
import sys
import unittest

import pandas as pd

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

from time_splitter import (  # noqa: E402
    create_time_splits,
    filter_and_sample_splits,
    get_split_info,
)


def create_synthetic_data(start_date='2020-01-01', periods=36, freq='ME'):
    """
    Create a synthetic dataset for testing.

    Args:
        start_date: Start date for the date range
        periods: Number of periods (rows) to generate
        freq: Pandas frequency string. 'ME' means 'Month End' frequency,
              generating dates at the end of each month.
    """
    dates = pd.date_range(start_date, periods=periods, freq=freq)
    df = pd.DataFrame({
        'DATE': dates,
        'FIGHTER': ['A', 'B', 'C', 'D'] * (periods // 4) + ['A'] * (periods % 4),
        'opp_FIGHTER': ['B', 'A', 'D', 'C'] * (periods // 4) + ['B'] * (periods % 4),
        'result': [1, 0, 1, 0] * (periods // 4) + [1] * (periods % 4),
        'value': range(periods),
    })
    return df


class TestFilterAndSampleSplits(unittest.TestCase):
    """Tests for filter_and_sample_splits function."""

    def setUp(self):
        """Create sample data and splits for testing."""
        # Create 3 years of monthly data
        self.df = create_synthetic_data(periods=36)
        # Create splits with 3-month validation periods
        self.splits = create_time_splits(self.df, months=3, min_train_months=3)

    def test_no_filtering_returns_all_splits(self):
        """Test that no filtering returns all splits unchanged."""
        filtered = filter_and_sample_splits(
            self.splits,
            min_val_size=0,
            max_splits=None
        )
        self.assertEqual(len(filtered), len(self.splits))

    def test_min_val_size_filters_small_splits(self):
        """Test that min_val_size correctly filters out small validation sets."""
        # Create splits where some have very small validation sets
        # by using a larger month window on smaller data
        small_df = create_synthetic_data(periods=24)
        splits = create_time_splits(small_df, months=6, min_train_months=3)

        if len(splits) > 0:
            # Get the minimum validation size
            min_val = min(len(val_df) for _, val_df in splits)
            max_val = max(len(val_df) for _, val_df in splits)

            # Filter with a threshold that should remove some splits
            threshold = min_val + 1 if max_val > min_val else min_val + 1
            filtered = filter_and_sample_splits(splits, min_val_size=threshold)

            # All remaining splits should meet the threshold
            for _, val_df in filtered:
                self.assertGreaterEqual(len(val_df), threshold)

    def test_min_val_size_of_zero_keeps_all(self):
        """Test that min_val_size=0 keeps all splits."""
        filtered = filter_and_sample_splits(self.splits, min_val_size=0)
        self.assertEqual(len(filtered), len(self.splits))

    def test_max_splits_limits_output(self):
        """Test that max_splits limits the number of returned splits."""
        if len(self.splits) > 3:
            filtered = filter_and_sample_splits(self.splits, min_val_size=0, max_splits=3)
            self.assertEqual(len(filtered), 3)

    def test_max_splits_none_keeps_all(self):
        """Test that max_splits=None keeps all splits after filtering."""
        filtered = filter_and_sample_splits(self.splits, min_val_size=0, max_splits=None)
        self.assertEqual(len(filtered), len(self.splits))

    def test_even_strategy_distributes_evenly(self):
        """Test that 'even' strategy produces evenly distributed splits."""
        if len(self.splits) >= 6:
            filtered = filter_and_sample_splits(
                self.splits,
                min_val_size=0,
                max_splits=3,
                sample_strategy="even"
            )
            self.assertEqual(len(filtered), 3)

            # Check that the selected splits are evenly distributed
            # (first, middle, last approximately)
            selected_dfs = [f[0] for f in filtered]
            original_dfs = [s[0] for s in self.splits]

            # Verify we got first and last
            # The even strategy should include the first split
            self.assertTrue(selected_dfs[0].equals(original_dfs[0]))

    def test_random_strategy_is_reproducible(self):
        """Test that 'random' strategy with same seed is reproducible."""
        if len(self.splits) >= 4:
            filtered1 = filter_and_sample_splits(
                self.splits,
                min_val_size=0,
                max_splits=2,
                sample_strategy="random",
                sample_seed=42
            )
            filtered2 = filter_and_sample_splits(
                self.splits,
                min_val_size=0,
                max_splits=2,
                sample_strategy="random",
                sample_seed=42
            )

            # Both should have same splits
            self.assertEqual(len(filtered1), len(filtered2))
            for (t1, v1), (t2, v2) in zip(filtered1, filtered2):
                self.assertTrue(t1.equals(t2))
                self.assertTrue(v1.equals(v2))

    def test_random_strategy_different_seeds_differ(self):
        """Test that 'random' strategy with different seeds gives different results."""
        # Use a larger dataset to ensure more splits and reduce collision probability
        large_df = create_synthetic_data(periods=60)
        large_splits = create_time_splits(large_df, months=3, min_train_months=3)

        if len(large_splits) >= 10:
            filtered1 = filter_and_sample_splits(
                large_splits,
                min_val_size=0,
                max_splits=3,
                sample_strategy="random",
                sample_seed=42
            )
            filtered2 = filter_and_sample_splits(
                large_splits,
                min_val_size=0,
                max_splits=3,
                sample_strategy="random",
                sample_seed=7
            )

            # Results should likely differ (not guaranteed but highly probable)
            # Compare validation start dates to check for differences
            val_dates1 = [v['DATE'].min() for _, v in filtered1]
            val_dates2 = [v['DATE'].min() for _, v in filtered2]
            # With different seeds on a larger set, the selected indices should differ
            self.assertNotEqual(val_dates1, val_dates2)

    def test_invalid_strategy_raises_error(self):
        """Test that invalid sample_strategy raises ValueError."""
        with self.assertRaises(ValueError):
            filter_and_sample_splits(self.splits, sample_strategy="invalid")

    def test_negative_min_val_size_raises_error(self):
        """Test that negative min_val_size raises ValueError."""
        with self.assertRaises(ValueError):
            filter_and_sample_splits(self.splits, min_val_size=-1)

    def test_negative_max_splits_raises_error(self):
        """Test that negative max_splits raises ValueError."""
        with self.assertRaises(ValueError):
            filter_and_sample_splits(self.splits, max_splits=-1)

    def test_max_splits_zero_returns_empty(self):
        """Test that max_splits=0 returns empty list."""
        filtered = filter_and_sample_splits(self.splits, max_splits=0)
        self.assertEqual(len(filtered), 0)

    def test_empty_splits_returns_empty(self):
        """Test that empty input returns empty output."""
        filtered = filter_and_sample_splits([])
        self.assertEqual(len(filtered), 0)

    def test_filtering_preserves_tuple_structure(self):
        """Test that filtered splits maintain (train_df, val_df) structure."""
        filtered = filter_and_sample_splits(self.splits, min_val_size=0, max_splits=2)

        for split in filtered:
            self.assertIsInstance(split, tuple)
            self.assertEqual(len(split), 2)
            train_df, val_df = split
            self.assertIsInstance(train_df, pd.DataFrame)
            self.assertIsInstance(val_df, pd.DataFrame)

    def test_max_splits_greater_than_available(self):
        """Test that max_splits > available splits returns all available."""
        filtered = filter_and_sample_splits(self.splits, min_val_size=0, max_splits=1000)
        self.assertEqual(len(filtered), len(self.splits))


class TestCreateTimeSplitsWithFiltering(unittest.TestCase):
    """Integration tests for create_time_splits with filtering."""

    def setUp(self):
        """Create sample data for testing."""
        self.df = create_synthetic_data(periods=48)

    def test_end_to_end_create_and_filter(self):
        """Test creating splits and filtering them."""
        # Create splits
        splits = create_time_splits(self.df, months=6, min_train_months=6)

        # Filter
        filtered = filter_and_sample_splits(
            splits,
            min_val_size=2,
            max_splits=3,
            sample_strategy="even"
        )

        # Should have at most 3 splits
        self.assertLessEqual(len(filtered), 3)

        # Each validation set should have at least 2 rows
        for _, val_df in filtered:
            self.assertGreaterEqual(len(val_df), 2)

    def test_filter_then_get_info(self):
        """Test that get_split_info works on filtered splits."""
        splits = create_time_splits(self.df, months=6, min_train_months=6)
        filtered = filter_and_sample_splits(splits, max_splits=2)

        info = get_split_info(filtered)

        self.assertEqual(len(info), len(filtered))
        self.assertIn('split_idx', info.columns)
        self.assertIn('val_rows', info.columns)


class TestFilteringDeterminism(unittest.TestCase):
    """Tests for deterministic behavior of filtering."""

    def test_even_strategy_is_deterministic(self):
        """Test that 'even' strategy always returns same result."""
        df = create_synthetic_data(periods=36)
        splits = create_time_splits(df, months=3, min_train_months=3)

        results = []
        for _ in range(5):
            filtered = filter_and_sample_splits(
                splits,
                max_splits=3,
                sample_strategy="even"
            )
            results.append([(len(t), len(v)) for t, v in filtered])

        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)

    def test_random_strategy_with_seed_is_deterministic(self):
        """Test that 'random' strategy with seed is deterministic."""
        df = create_synthetic_data(periods=36)
        splits = create_time_splits(df, months=3, min_train_months=3)

        results = []
        for _ in range(5):
            filtered = filter_and_sample_splits(
                splits,
                max_splits=3,
                sample_strategy="random",
                sample_seed=42
            )
            results.append([(len(t), len(v)) for t, v in filtered])

        # All results should be identical
        for result in results[1:]:
            self.assertEqual(results[0], result)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases in filtering."""

    def test_single_split(self):
        """Test filtering with a single split."""
        df = create_synthetic_data(periods=24)
        splits = create_time_splits(df, months=12, min_train_months=6)

        if len(splits) >= 1:
            # Take just one split
            single_split = splits[:1]

            # Filter should work
            filtered = filter_and_sample_splits(single_split, max_splits=1)
            self.assertEqual(len(filtered), 1)

    def test_high_min_val_size_filters_all(self):
        """Test that very high min_val_size can filter out all splits."""
        df = create_synthetic_data(periods=24)
        splits = create_time_splits(df, months=6, min_train_months=3)

        # Use a very high threshold
        filtered = filter_and_sample_splits(splits, min_val_size=1000)
        self.assertEqual(len(filtered), 0)

    def test_filtering_maintains_chronological_order(self):
        """Test that filtered splits maintain chronological order."""
        df = create_synthetic_data(periods=48)
        splits = create_time_splits(df, months=3, min_train_months=3)

        filtered = filter_and_sample_splits(splits, max_splits=5, sample_strategy="random", sample_seed=42)

        if len(filtered) > 1:
            # Check that validation start dates are in order
            val_starts = [val_df['DATE'].min() for _, val_df in filtered]
            self.assertEqual(val_starts, sorted(val_starts))


if __name__ == '__main__':
    unittest.main(verbosity=2)
