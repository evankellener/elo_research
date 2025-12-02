"""
Time-based splitter for reproducible validation pipeline.

This module provides utilities for creating non-overlapping, contiguous time-splits
from historical match data. It supports both expanding and rolling window strategies
for training/validation splits.
"""

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
    # Convert string to DateOffset if needed
    if isinstance(offset, str):
        offset = pd.DateOffset(months=_parse_offset_months(offset))
    
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
