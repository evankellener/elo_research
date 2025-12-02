#!/bin/bash
# Quick example run script for GA testing
# Runs a small GA with train/val split for quick sanity checking
#
# Usage:
#   ./run_example_quick.sh           # Run with defaults
#   ./run_example_quick.sh --seed 42 # Run with specific seed
#
# This script runs:
#   - 3 generations (very quick)
#   - Population of 5
#   - 50% train/validation split
#   - Outputs JSON results
#
# Expected runtime: ~1-2 minutes

set -e

# Change to scripts directory (relative to this script's location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
SEED=${SEED:-42}
GENERATIONS=${GENERATIONS:-3}
POPULATION=${POPULATION:-5}
LOOKBACK=${LOOKBACK:-365}
TRAIN_VAL_SPLIT=${TRAIN_VAL_SPLIT:-"50%"}
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ga_quick_test}"

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --generations)
            GENERATIONS="$2"
            shift 2
            ;;
        --population)
            POPULATION="$2"
            shift 2
            ;;
        --lookback)
            LOOKBACK="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set output paths
OUTPUT_JSON="$OUTPUT_DIR/quick_test_results.json"
OUTPUT_LOG="$OUTPUT_DIR/quick_test.log"

echo "=== GA Quick Example Run ==="
echo "  Seed: $SEED"
echo "  Generations: $GENERATIONS"
echo "  Population: $POPULATION"
echo "  Lookback days: $LOOKBACK"
echo "  Train/Val split: $TRAIN_VAL_SPLIT"
echo "  Output JSON: $OUTPUT_JSON"
echo "  Output log: $OUTPUT_LOG"
echo ""

# Run the GA
python full_genetic_with_k_denom_mov.py \
    --mode roi \
    --seed "$SEED" \
    --generations "$GENERATIONS" \
    --population "$POPULATION" \
    --lookback-days "$LOOKBACK" \
    --train-val-split "$TRAIN_VAL_SPLIT" \
    --output-json "$OUTPUT_JSON" \
    --show-comprehensive \
    2>&1 | tee "$OUTPUT_LOG"

# Check if successful
if [ -f "$OUTPUT_JSON" ]; then
    echo ""
    echo "=== Quick Test Completed Successfully ==="
    echo "Results saved to: $OUTPUT_JSON"
    echo ""
    
    # Print key metrics from JSON
    python -c "
import json
with open('$OUTPUT_JSON') as f:
    data = json.load(f)
    
    print('=== Key Metrics ===')
    summary = data.get('summary', {})
    print(f\"  ROI: {summary.get('roi_percent', 'N/A')}%\")
    print(f\"  Accuracy: {float(summary.get('accuracy', 0))*100:.1f}%\" if summary.get('accuracy') else '  Accuracy: N/A')
    print(f\"  Win Rate: {float(summary.get('win_rate', 0))*100:.0f}%\" if summary.get('win_rate') else '  Win Rate: N/A')
    print(f\"  ECE: {summary.get('ece', 'N/A')}\")
    
    # Train/Val metrics
    best_metrics = data.get('best_metrics', {})
    if best_metrics.get('train'):
        print('')
        print('=== Train/Val/OOS Split ===')
        train = best_metrics.get('train', {})
        val = best_metrics.get('val', {})
        oos = best_metrics.get('oos', {})
        print(f\"  Train ROI: {train.get('roi_percent', 'N/A')}%\")
        print(f\"  Val ROI: {val.get('roi_percent', 'N/A')}%\")
        if oos:
            print(f\"  OOS ROI: {oos.get('roi_percent', 'N/A')}%\")
"
else
    echo ""
    echo "=== Quick Test Failed ==="
    echo "No output JSON file created"
    exit 1
fi
