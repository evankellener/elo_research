"""
Time-based splitter for reproducible validation pipeline.

This module provides utilities for creating non-overlapping, contiguous time-splits
from historical match data. It supports both expanding and rolling window strategies
for training/validation splits.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
from dateutil.relativedelta import relativedelta


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect the time/date column in a DataFrame.

    Checks for common date column names: DATE, date, timestamp, Date, Timestamp.

    Args:
        df: Input DataFrame

    Returns:
        str: Name of detected time column, or None if not found
    """
    common_names = ['DATE', 'date', 'timestamp', 'Timestamp', 'Date', 'TIMESTAMP']

    for name in common_names:
        if name in df.columns:
            return name

    return None


def create_time_splits(
    df: pd.DataFrame,
    months: int = 6,
    time_column: Optional[str] = None,
    window_type: str = "expanding",
    min_train_months: int = 6
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create non-overlapping contiguous time-splits from historical matches dataset.

    Each split consists of a training DataFrame and a validation DataFrame.
    The validation fold is a contiguous period of the specified duration.

    Args:
        df: DataFrame with historical match data. Must contain a date/time column.
        months: Duration of each validation split in months (e.g., 2, 6, 12).
                Can also use pandas DateOffset strings via the offset parameter.
        time_column: Name of the timestamp/date column. If None, auto-detects
                    by looking for 'DATE', 'date', or 'timestamp' columns.
        window_type: Strategy for building training data:
                    - "expanding": Training uses ALL data before the validation fold
                      (expanding window grows with each split)
                    - "rolling": Training uses a fixed window immediately before
                      the validation fold (same size as validation, i.e., `months`)
        min_train_months: Minimum number of months required in training data.
                         Splits that would have less training data are skipped.

    Returns:
        List of (train_df, val_df) tuples. Each tuple contains:
        - train_df: Training data (rows before the validation period)
        - val_df: Validation data (rows in the validation period)

        The last split may be shorter than `months` if there isn't enough
        data remaining.

    Raises:
        ValueError: If time_column cannot be auto-detected or doesn't exist.

    Example:
        >>> splits = create_time_splits(df, months=6, window_type="expanding")
        >>> for train_df, val_df in splits:
        ...     # Train model on train_df, evaluate on val_df
        ...     model.fit(train_df)
        ...     roi = model.evaluate(val_df)

    Notes:
        - The function ensures chronological ordering and non-overlapping splits.
        - For "expanding" window: training always starts from the beginning of data.
        - For "rolling" window: training window size equals validation window size.
        - Minimum training data requirement helps ensure statistical validity.
    """
    # Auto-detect time column if not specified
    if time_column is None:
        time_column = detect_time_column(df)
        if time_column is None:
            raise ValueError(
                "Could not auto-detect time column. Please specify time_column parameter. "
                "Common names checked: DATE, date, timestamp"
            )

    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame. "
                        f"Available columns: {list(df.columns)[:10]}...")

    if window_type not in ("expanding", "rolling"):
        raise ValueError(f"window_type must be 'expanding' or 'rolling', got '{window_type}'")

    # Make a copy and ensure date column is datetime
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])

    # Sort by date
    df = df.sort_values(time_column).reset_index(drop=True)

    # Get date range
    min_date = df[time_column].min()
    max_date = df[time_column].max()

    # Calculate minimum training start date (to ensure min_train_months of data)
    min_train_start = min_date + relativedelta(months=min_train_months)

    # Generate split boundaries
    splits = []

    # First validation period starts after minimum training period
    val_start = min_train_start

    while val_start < max_date:
        # Calculate validation end date
        val_end = val_start + relativedelta(months=months)

        # Validation data: rows in [val_start, val_end)
        val_mask = (df[time_column] >= val_start) & (df[time_column] < val_end)
        val_df = df[val_mask].copy()

        # Skip if no validation data
        if len(val_df) == 0:
            val_start = val_end
            continue

        # Training data depends on window_type
        if window_type == "expanding":
            # All data before validation start
            train_mask = df[time_column] < val_start
        else:  # rolling
            # Fixed window: same duration as validation, immediately before
            train_start = val_start - relativedelta(months=months)
            train_mask = (df[time_column] >= train_start) & (df[time_column] < val_start)

        train_df = df[train_mask].copy()

        # Skip if not enough training data
        if len(train_df) < 10:  # Minimum number of rows
            val_start = val_end
            continue

        splits.append((train_df, val_df))

        # Move to next validation period
        val_start = val_end

    return splits


def filter_and_sample_splits(
    splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
    min_val_size: int = 5,
    max_splits: Optional[int] = None,
    sample_strategy: str = "even",
    sample_seed: int = 42,
    time_column: str = "DATE"
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Filter and downsample time-based validation splits.

    This function provides deterministic filtering and downsampling of splits to:
    - Remove splits with too few validation samples (ensures statistical reliability)
    - Limit the total number of splits for computational efficiency

    Args:
        splits: List of (train_df, val_df) tuples from create_time_splits()
        min_val_size: Minimum number of rows required in validation set.
                     Splits with fewer validation samples are removed.
                     Default: 5
        max_splits: Maximum number of splits to keep after filtering.
                   If None, all filtered splits are kept.
                   Default: None
        sample_strategy: Strategy for selecting which splits to keep when
                        max_splits < number of filtered splits:
                        - "even": Evenly spaced splits across time (default)
                        - "random": Random selection with seed
        sample_seed: Random seed for reproducibility when strategy="random".
                    Default: 42
        time_column: Name of the date column for logging purposes.
                    Default: "DATE"

    Returns:
        List of (train_df, val_df) tuples after filtering and sampling.
        Returns empty list if all splits are filtered out.

    Example:
        >>> splits = create_time_splits(df, months=6)
        >>> # Keep only splits with at least 10 validation samples
        >>> filtered = filter_and_sample_splits(splits, min_val_size=10)
        >>> # Keep at most 5 evenly-spaced splits
        >>> sampled = filter_and_sample_splits(splits, max_splits=5, sample_strategy="even")

    Notes:
        - Filtering by min_val_size happens before sampling by max_splits.
        - The "even" strategy preserves temporal coverage across the data.
        - The "random" strategy is reproducible with the same seed.
    """
    import random as py_random

    if sample_strategy not in ("even", "random"):
        raise ValueError(f"sample_strategy must be 'even' or 'random', got '{sample_strategy}'")

    if min_val_size < 0:
        raise ValueError(f"min_val_size must be non-negative, got {min_val_size}")

    if max_splits is not None and max_splits < 0:
        raise ValueError(f"max_splits must be non-negative, got {max_splits}")

    # Step 1: Filter by minimum validation size
    filtered_splits = [
        (train_df, val_df)
        for train_df, val_df in splits
        if len(val_df) >= min_val_size
    ]

    # Step 2: Sample if needed
    if max_splits is None or len(filtered_splits) <= max_splits:
        return filtered_splits

    if max_splits == 0:
        return []

    if sample_strategy == "even":
        # Evenly spaced indices across the filtered splits
        n = len(filtered_splits)
        # Use linspace to get evenly distributed indices
        indices = [int(round(i)) for i in np.linspace(0, n - 1, max_splits)]
        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)
        sampled_splits = [filtered_splits[i] for i in unique_indices]
    else:  # random
        rng = py_random.Random(sample_seed)
        indices = list(range(len(filtered_splits)))
        sampled_indices = sorted(rng.sample(indices, max_splits))
        sampled_splits = [filtered_splits[i] for i in sampled_indices]

    return sampled_splits


def create_time_splits_with_offset(
    df: pd.DataFrame,
    offset: Union[str, pd.DateOffset],
    time_column: Optional[str] = None,
    window_type: str = "expanding",
    min_train_periods: int = 1
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time splits using pandas DateOffset strings for more flexible durations.

    This is an alternative to create_time_splits() that accepts pandas DateOffset
    strings like '2M', '6M', '1Y', '3W' for more flexible split durations.

    Args:
        df: DataFrame with historical match data.
        offset: Split duration as pandas DateOffset or string (e.g., '6M', '2M', '1Y').
                Common values:
                - '2M' = 2 months
                - '6M' = 6 months
                - '12M' or '1Y' = 1 year
                - '3M' = 1 quarter
        time_column: Name of the timestamp column. Auto-detected if None.
        window_type: "expanding" or "rolling" window strategy.
        min_train_periods: Minimum number of offset periods required in training.

    Returns:
        List of (train_df, val_df) tuples.

    Example:
        >>> # Create 6-month splits
        >>> splits = create_time_splits_with_offset(df, '6M')
        >>> # Create quarterly splits with rolling window
        >>> splits = create_time_splits_with_offset(df, '3M', window_type='rolling')
    """
    # Convert string to DateOffset if needed using relativedelta for consistency
    if isinstance(offset, str):
        months = _parse_offset_months(offset)
        # Use relativedelta for consistency with the rest of the module
        # This handles large month values correctly
        offset = relativedelta(months=months)

    # Auto-detect time column
    if time_column is None:
        time_column = detect_time_column(df)
        if time_column is None:
            raise ValueError(
                "Could not auto-detect time column. Please specify time_column parameter."
            )

    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame.")

    if window_type not in ("expanding", "rolling"):
        raise ValueError(f"window_type must be 'expanding' or 'rolling', got '{window_type}'")

    # Make a copy and ensure date column is datetime
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column).reset_index(drop=True)

    min_date = df[time_column].min()
    max_date = df[time_column].max()

    # Start validation after minimum training period
    min_train_offset = offset * min_train_periods
    val_start = min_date + min_train_offset

    splits = []

    while val_start < max_date:
        val_end = val_start + offset

        # Validation data
        val_mask = (df[time_column] >= val_start) & (df[time_column] < val_end)
        val_df = df[val_mask].copy()

        if len(val_df) == 0:
            val_start = val_end
            continue

        # Training data
        if window_type == "expanding":
            train_mask = df[time_column] < val_start
        else:  # rolling
            train_start = val_start - offset
            train_mask = (df[time_column] >= train_start) & (df[time_column] < val_start)

        train_df = df[train_mask].copy()

        if len(train_df) < 10:
            val_start = val_end
            continue

        splits.append((train_df, val_df))
        val_start = val_end

    return splits


def _parse_offset_months(offset_str: str) -> int:
    """
    Parse a pandas-style offset string to number of months.

    Args:
        offset_str: Offset string like '2M', '6M', '12M', '1Y'

    Returns:
        Number of months as integer

    Raises:
        ValueError: If offset string cannot be parsed
    """
    offset_str = offset_str.strip().upper()

    if offset_str.endswith('M'):
        try:
            return int(offset_str[:-1])
        except ValueError:
            raise ValueError(f"Invalid month offset: {offset_str}")
    elif offset_str.endswith('Y'):
        try:
            return int(offset_str[:-1]) * 12
        except ValueError:
            raise ValueError(f"Invalid year offset: {offset_str}")
    else:
        raise ValueError(f"Unsupported offset format: {offset_str}. Use 'NM' for months or 'NY' for years.")


def get_split_info(splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
                   time_column: str = 'DATE') -> pd.DataFrame:
    """
    Get summary information about time splits.

    Args:
        splits: List of (train_df, val_df) tuples from create_time_splits()
        time_column: Name of the date column

    Returns:
        DataFrame with columns:
        - split_idx: Split index (0, 1, 2, ...)
        - train_start: Start date of training data
        - train_end: End date of training data
        - train_rows: Number of rows in training data
        - val_start: Start date of validation data
        - val_end: End date of validation data
        - val_rows: Number of rows in validation data
    """
    info = []

    for i, (train_df, val_df) in enumerate(splits):
        train_dates = train_df[time_column]
        val_dates = val_df[time_column]

        info.append({
            'split_idx': i,
            'train_start': train_dates.min(),
            'train_end': train_dates.max(),
            'train_rows': len(train_df),
            'val_start': val_dates.min(),
            'val_end': val_dates.max(),
            'val_rows': len(val_df),
        })

    return pd.DataFrame(info)
