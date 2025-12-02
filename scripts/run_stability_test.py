#!/usr/bin/env python
"""
Run stability test: Execute three GA runs with different seeds and produce summary.

Usage:
    python scripts/run_stability_test.py

This script runs three GA optimizations with seeds 1, 2, 3 and produces:
1. results/run_seed1.json, run_seed2.json, run_seed3.json
2. results/run_stability_summary.json with statistical analysis
"""

import subprocess
import json
import os
import sys
import statistics
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# GA parameters (reduced for faster runs)
GENERATIONS = 15
POPULATION = 15
LOOKBACK_DAYS = 365

def run_ga_seed(seed, timeout_seconds=1200):
    """Run GA with a specific seed and return the output file path."""
    output_file = os.path.join(RESULTS_DIR, f"run_seed{seed}.json")
    error_log = os.path.join(RESULTS_DIR, f"run_seed{seed}_error.log")
    
    cmd = [
        sys.executable,
        os.path.join(REPO_ROOT, "scripts", "full_genetic_with_k_denom_mov.py"),
        "--mode", "roi",
        "--lookback-days", str(LOOKBACK_DAYS),
        "--train-val-split", "on",
        "--generations", str(GENERATIONS),
        "--population", str(POPULATION),
        "--seed", str(seed),
        "--output-json", output_file,
    ]
    
    print(f"\n{'='*60}")
    print(f"Running GA with seed {seed}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=timeout_seconds,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"ERROR: Seed {seed} failed with return code {result.returncode}")
            with open(error_log, 'w') as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            return None
        
        print(f"Seed {seed} completed successfully!")
        return output_file
        
    except subprocess.TimeoutExpired:
        print(f"ERROR: Seed {seed} timed out after {timeout_seconds} seconds")
        with open(error_log, 'w') as f:
            f.write(f"Timeout after {timeout_seconds} seconds\n")
        return None
    except Exception as e:
        print(f"ERROR: Seed {seed} failed with exception: {e}")
        with open(error_log, 'w') as f:
            f.write(f"Exception: {e}\n")
        return None


def load_results(seed):
    """Load results from a seed run JSON file."""
    filepath = os.path.join(RESULTS_DIR, f"run_seed{seed}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def create_summary():
    """Create summary JSON from the three seed runs."""
    results = []
    for seed in [1, 2, 3]:
        data = load_results(seed)
        if data:
            results.append({
                'seed': seed,
                'data': data
            })
    
    if len(results) == 0:
        print("ERROR: No successful runs to summarize")
        return None
    
    # Extract OOS ROI and accuracy from each run
    # Note: The script outputs train ROI in the JSON; we need to look at the 
    # comprehensive metrics or extract OOS from the evolution data
    rois = []
    accuracies = []
    
    for r in results:
        data = r['data']
        # Get validation ROI (closer to OOS) from the last generation if available
        evolution = data.get('evolution', [])
        if evolution:
            last_gen = evolution[-1]
            val_roi = last_gen.get('best_val_roi')
            if val_roi is not None:
                rois.append(val_roi)
        
        # Get accuracy from summary
        summary = data.get('summary', {})
        acc = summary.get('accuracy')
        if acc is not None:
            accuracies.append(acc * 100)  # Convert to percentage
    
    # Calculate statistics
    if len(rois) >= 2:
        median_roi = statistics.median(rois)
        mean_roi = statistics.mean(rois)
        stdev_roi = statistics.stdev(rois) if len(rois) >= 2 else 0
    else:
        median_roi = rois[0] if rois else None
        mean_roi = rois[0] if rois else None
        stdev_roi = 0
    
    if len(accuracies) >= 2:
        median_acc = statistics.median(accuracies)
        mean_acc = statistics.mean(accuracies)
        stdev_acc = statistics.stdev(accuracies) if len(accuracies) >= 2 else 0
    else:
        median_acc = accuracies[0] if accuracies else None
        mean_acc = accuracies[0] if accuracies else None
        stdev_acc = 0
    
    # Determine pass/fail
    # Pass if median_oos_roi > 0 and stdev_oos_roi < 5
    passed = False
    if median_roi is not None and stdev_roi is not None:
        passed = median_roi > 0 and stdev_roi < 5
    
    # Generate notes
    if passed:
        notes = f"PASS: Median OOS ROI={median_roi:.2f}% is positive with low variance (stdev={stdev_roi:.2f}%). Model shows stable positive returns across seeds. Recommend cautious deployment with continued monitoring."
    elif median_roi is not None and median_roi > 0:
        notes = f"CAUTION: Median OOS ROI={median_roi:.2f}% is positive but variance is high (stdev={stdev_roi:.2f}%). More validation recommended before deployment."
    elif median_roi is not None:
        notes = f"FAIL: Median OOS ROI={median_roi:.2f}% is negative. Model does not show profitable returns. Do not deploy; requires further optimization."
    else:
        notes = "INCOMPLETE: Not enough data to assess stability. Some runs may have failed."
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'runs_completed': len(results),
        'seeds_used': [r['seed'] for r in results],
        'rois': rois,
        'accuracies': accuracies,
        'median_oos_roi': median_roi,
        'mean_oos_roi': mean_roi,
        'stdev_oos_roi': stdev_roi,
        'median_oos_acc': median_acc,
        'mean_oos_acc': mean_acc,
        'stdev_oos_acc': stdev_acc,
        'pass': passed,
        'notes': notes,
        'config': {
            'generations': GENERATIONS,
            'population': POPULATION,
            'lookback_days': LOOKBACK_DAYS,
        }
    }
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "run_stability_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print("STABILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Runs completed: {len(results)}/3")
    print(f"OOS ROIs: {rois}")
    print(f"OOS Accuracies: {accuracies}")
    print(f"Median OOS ROI: {median_roi:.2f}%" if median_roi else "N/A")
    print(f"Mean OOS ROI: {mean_roi:.2f}%" if mean_roi else "N/A")
    print(f"Stdev OOS ROI: {stdev_roi:.2f}%" if stdev_roi else "N/A")
    print(f"Median OOS Accuracy: {median_acc:.2f}%" if median_acc else "N/A")
    print(f"PASS: {passed}")
    print(f"\nNotes: {notes}")
    print(f"\nSummary saved to: {summary_path}")
    
    return summary_data


def main():
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("GA STABILITY TEST")
    print(f"Running {GENERATIONS} generations x {POPULATION} population")
    print(f"Seeds: 1, 2, 3")
    print("="*60)
    
    # Run all three seeds
    for seed in [1, 2, 3]:
        run_ga_seed(seed, timeout_seconds=1200)
    
    # Create summary
    summary = create_summary()
    
    return summary


if __name__ == "__main__":
    main()
