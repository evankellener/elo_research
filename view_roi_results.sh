#!/bin/bash
# Quick script to view ROI optimization results

echo "=== ROI Optimization Results ==="
echo ""

# Check if script is still running
if pgrep -f "ga_time_split_roi.py" > /dev/null; then
    echo "⚠️  Script is still running..."
    echo "Current progress:"
    tail -5 nohup.out
    echo ""
    echo "To watch live: tail -f nohup.out"
    echo ""
else
    echo "✓ Script has completed"
    echo ""
fi

# Show best parameters
if [ -f "ga_roi_best_params.txt" ]; then
    echo "=== Best Parameters ==="
    cat ga_roi_best_params.txt
    echo ""
fi

# Show final output
if [ -f "nohup.out" ]; then
    echo "=== Final Results from Console ==="
    tail -20 nohup.out
    echo ""
fi

# List generated files
echo "=== Generated Files ==="
ls -lh ga_roi*.{csv,txt} 2>/dev/null | awk '{print $9, "(" $5 ")"}'
ls -lh images/ga_roi*.png 2>/dev/null | awk '{print $9, "(" $5 ")"}'

echo ""
echo "To view the plot: open images/ga_roi_results.png"
echo "To analyze history: python -c \"import pandas as pd; df=pd.read_csv('ga_roi_history.csv'); print(df.tail(10))\""
