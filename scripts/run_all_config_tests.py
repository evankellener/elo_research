#!/usr/bin/env python3
"""
Run All Configuration Tests

This script runs the genetic algorithm with all 8 combinations of feature flags:
- multiphase-decay (on/off)
- weight-adjust (on/off)
- optimize-elo-denom (on/off)

It collects ROI and accuracy metrics from each run and generates a comparison report.

Features:
- Reproducible results with seed system
- Proper subprocess output parsing with debug logging
- None value handling for partial data
- Comprehensive comparison report
- Real-time progress tracking with generation output
- Per-iteration detailed summaries with timing
- Running comparison against baseline/best configuration

Usage:
    python scripts/run_all_config_tests.py
    python scripts/run_all_config_tests.py --seed 42
    python scripts/run_all_config_tests.py --generations 5 --population 10
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime


def format_elapsed_time(seconds):
    """Format elapsed time in human-readable format (e.g., '5m 42s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs:02d}s"


def format_flag(value):
    """Format a feature flag value with checkmark symbols."""
    return "✓" if value == 'on' else "✗"


def format_roi_with_sign(value):
    """Format ROI value with explicit +/- sign."""
    if value is None:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_roi_comparison(current_roi, baseline_roi):
    """Format ROI comparison against baseline."""
    if current_roi is None or baseline_roi is None:
        return ""
    diff = current_roi - baseline_roi
    if abs(diff) < 0.01:
        return " (same as baseline)"
    sign = "↑" if diff > 0 else "↓"
    return f" ({sign}{abs(diff):.2f}% vs baseline)"


def parse_subprocess_output(output):
    """
    Parse subprocess output to extract ROI, accuracy, and best parameters.
    
    Args:
        output: String output from subprocess
        
    Returns:
        dict: Contains 'oos_roi', 'oos_accuracy', 'best_params', 'train_roi', 
              'num_bets', 'wins', 'win_rate', 'total_wagered', 'total_profit' (may be None if not found)
    """
    result = {
        'oos_roi': None,
        'oos_accuracy': None,
        'best_params': None,
        'train_roi': None,
        'num_bets': None,
        'wins': None,
        'win_rate': None,
        'total_wagered': None,
        'total_profit': None,
        'raw_output': output
    }
    
    if not output:
        print("  [DEBUG] No output to parse")
        return result
    
    # Try to find OOS ROI patterns
    # Pattern 1: "ROI: X.XX%" or "ROI: +X.XX%" or "ROI: -X.XX%"
    roi_patterns = [
        r'(?:OOS\s+)?ROI[:\s]+([+-]?\d+\.?\d*)%',
        r'roi_percent[:\s]+([+-]?\d+\.?\d*)',
        r"'roi_percent'[:\s]+([+-]?\d+\.?\d*)",
        r'OOS\s+ROI[:\s]+([+-]?\d+\.?\d*)%',
        r'Best ROI[^:]*:[:\s]+([+-]?\d+\.?\d*)%',
    ]
    
    for pattern in roi_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                result['oos_roi'] = float(match.group(1))
                print(f"  [DEBUG] Found ROI with pattern '{pattern}': {result['oos_roi']}")
                break
            except (ValueError, IndexError):
                continue
    
    # Try to find OOS Accuracy patterns
    # Pattern: "Accuracy: X.XX%" or "OOS accuracy: X.XX"
    accuracy_patterns = [
        r'(?:OOS\s+)?[Aa]ccuracy[:\s]+(\d+\.?\d*)%',
        r'(?:OOS\s+)?[Aa]ccuracy[:\s]+(\d+\.?\d*)',
        r'Overall OOS accuracy[:\s]+(\d+\.?\d*)',
        r"'accuracy'[:\s]+(\d+\.?\d*)",
    ]
    
    for pattern in accuracy_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                # If value is less than 1, it's already a decimal
                # If value is greater than 1, it's a percentage
                if value > 1:
                    value = value / 100.0
                result['oos_accuracy'] = value
                print(f"  [DEBUG] Found Accuracy with pattern '{pattern}': {result['oos_accuracy']}")
                break
            except (ValueError, IndexError):
                continue
    
    # Try to extract best parameters
    # Look for patterns like: {'k': 123.45, 'w_ko': 1.23, ...}
    best_params = {}
    
    # Pattern for individual parameters - capture full decimal precision
    # Base parameters (always present)
    param_patterns = {
        'k': r"'k'[:\s]+(\d+(?:\.\d+)?)",
        'w_ko': r"'w_ko'[:\s]+(\d+(?:\.\d+)?)",
        'w_sub': r"'w_sub'[:\s]+(\d+(?:\.\d+)?)",
        'w_udec': r"'w_udec'[:\s]+(\d+(?:\.\d+)?)",
        'w_sdec': r"'w_sdec'[:\s]+(\d+(?:\.\d+)?)",
        'w_mdec': r"'w_mdec'[:\s]+(\d+(?:\.\d+)?)",
        # Multiphase decay parameters (if enabled)
        'quick_succession_days': r"'quick_succession_days'[:\s]+(\d+(?:\.\d+)?)",
        'quick_succession_bump': r"'quick_succession_bump'[:\s]+(\d+(?:\.\d+)?)",
        'decay_days': r"'decay_days'[:\s]+(\d+(?:\.\d+)?)",
        'multiphase_decay_rate': r"'multiphase_decay_rate'[:\s]+(\d+(?:\.\d+)?)",
        # Weight adjustment parameters (if enabled)
        'weight_up_precomp_penalty': r"'weight_up_precomp_penalty'[:\s]+(\d+(?:\.\d+)?)",
        'weight_up_postcomp_bonus': r"'weight_up_postcomp_bonus'[:\s]+(\d+(?:\.\d+)?)",
        # Elo denominator (if enabled)
        'elo_denom': r"'elo_denom'[:\s]+(\d+(?:\.\d+)?)",
        # Legacy decay parameters
        'decay_rate': r"'decay_rate'[:\s]+(\d+(?:\.\d+)?)",
        'min_days': r"'min_days'[:\s]+(\d+(?:\.\d+)?)",
    }
    
    for param_name, pattern in param_patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                best_params[param_name] = float(match.group(1))
            except (ValueError, IndexError):
                continue
    
    # If we found at least k and some weights, store them
    if 'k' in best_params and len(best_params) >= 4:
        result['best_params'] = best_params
        print(f"  [DEBUG] Found best params: {len(best_params)} parameters captured")
    
    # Extract training ROI (best ROI during GA optimization)
    train_roi_patterns = [
        r'Best ROI on.*?:\s*([+-]?\d+(?:\.\d+)?)%',
        r'best ROI=([+-]?\d+(?:\.\d+)?)%',
        r'\*\*\* NEW BEST:.*ROI=([+-]?\d+(?:\.\d+)?)%',
    ]
    for pattern in train_roi_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            try:
                # Take the last match as the best training ROI
                result['train_roi'] = float(matches[-1])
                break
            except (ValueError, IndexError):
                continue
    
    # Extract OOS metrics: Total Bets, Wins, Accuracy
    bets_match = re.search(r'Total Bets:\s*(\d+)', output)
    if bets_match:
        result['num_bets'] = int(bets_match.group(1))
    
    wins_match = re.search(r'Wins:\s*(\d+)', output)
    if wins_match:
        result['wins'] = int(wins_match.group(1))
    
    # Calculate win rate if we have both num_bets and wins (including 0 values)
    if result['num_bets'] is not None and result['wins'] is not None:
        result['win_rate'] = result['wins'] / result['num_bets'] if result['num_bets'] > 0 else 0
    
    # Extract Total Wagered and Total Profit
    wagered_match = re.search(r'Total Wagered:\s*\$?(\d+(?:\.\d+)?)', output)
    if wagered_match:
        result['total_wagered'] = float(wagered_match.group(1))
    
    profit_match = re.search(r'Total Profit:\s*\$?([+-]?\d+(?:\.\d+)?)', output)
    if profit_match:
        result['total_profit'] = float(profit_match.group(1))
    
    if result['oos_roi'] is None:
        print("  [DEBUG] Could not extract OOS ROI from output")
        # Print last few lines for debugging
        lines = output.strip().split('\n')
        print("  [DEBUG] Last 10 lines of output:")
        for line in lines[-10:]:
            print(f"    {line}")
    
    if result['oos_accuracy'] is None:
        print("  [DEBUG] Could not extract OOS Accuracy from output")
    
    return result


def convert_to_percentage(value):
    """
    Convert a value to percentage format.
    
    If value is <= 1, assumes it's a decimal (e.g., 0.573) and multiplies by 100.
    If value is > 1, assumes it's already a percentage (e.g., 57.3).
    
    Args:
        value: Numeric value that may be a decimal or percentage (can be None)
        
    Returns:
        float: Value as a percentage, or None if input is None
    """
    if value is None:
        return None
    return value * 100 if value <= 1 else value


def format_value(value, format_spec=".2f", prefix="", suffix=""):
    """
    Safely format a value, handling None values gracefully.
    
    Args:
        value: The value to format (can be None)
        format_spec: Format specifier (e.g., ".2f", ".1f", ".0f")
        prefix: String to prepend (e.g., "+")
        suffix: String to append (e.g., "%")
        
    Returns:
        str: Formatted string or "N/A" if value is None
    """
    if value is None:
        return "N/A"
    
    try:
        # Build the format string dynamically
        formatted = f"{value:{format_spec}}"
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError) as e:
        print(f"  [DEBUG] Format error for value {value} with spec {format_spec}: {e}")
        return "N/A"


def ensure_results_dir(script_dir):
    """
    Ensure the results directory exists.
    
    Args:
        script_dir: Path to the scripts directory
        
    Returns:
        str: Path to the results directory
    """
    results_dir = os.path.join(os.path.dirname(script_dir), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir


def save_params_to_json(config_name, params, metadata, results_dir):
    """
    Save full parameters to a JSON file for a single configuration.
    
    Args:
        config_name: Name of the configuration (e.g., 'baseline', 'md+wa')
        params: Dictionary of all parameters with full precision
        metadata: Dictionary with additional metadata (seed, generations, ROI, etc.)
        results_dir: Path to results directory
        
    Returns:
        str: Path to the saved JSON file
    """
    output_file = os.path.join(results_dir, f"{config_name}_best_params.json")
    
    output_data = {
        "config_name": config_name,
        "parameters": params,
        "metadata": metadata,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return output_file


def generate_python_dict_output(config_name, params):
    """
    Generate copy-paste ready Python dictionary string.
    
    Args:
        config_name: Name of the configuration
        params: Dictionary of all parameters
        
    Returns:
        str: Formatted Python dictionary string
    """
    lines = [f"# {config_name} config"]
    lines.append(f"params_{config_name.replace('+', '_').replace('-', '_')} = {{")
    
    # Define the order for base parameters
    base_params = ['k', 'w_ko', 'w_sub', 'w_udec', 'w_sdec', 'w_mdec']
    # Multiphase decay params
    md_params = ['quick_succession_days', 'quick_succession_bump', 'decay_days', 'multiphase_decay_rate']
    # Weight adjustment params
    wa_params = ['weight_up_precomp_penalty', 'weight_up_postcomp_bonus']
    # Elo denom param
    ed_params = ['elo_denom']
    # Legacy decay params
    decay_params = ['decay_rate', 'min_days']
    
    all_param_order = base_params + md_params + wa_params + ed_params + decay_params
    
    # Add parameters in order, only if they exist
    param_lines = []
    for param_name in all_param_order:
        if param_name in params:
            value = params[param_name]
            # Format with full precision
            if isinstance(value, float):
                param_lines.append(f'    "{param_name}": {repr(value)},')
            else:
                param_lines.append(f'    "{param_name}": {value},')
    
    lines.extend(param_lines)
    lines.append("}")
    
    return '\n'.join(lines)


def print_full_params_display(config_name, params, json_path=None):
    """
    Print full parameters in a formatted display.
    
    Args:
        config_name: Name of the configuration
        params: Dictionary of all parameters
        json_path: Optional path to the saved JSON file
    """
    # Define parameter groups
    base_params = ['k', 'w_ko', 'w_sub', 'w_udec', 'w_sdec', 'w_mdec']
    md_params = ['quick_succession_days', 'quick_succession_bump', 'decay_days', 'multiphase_decay_rate']
    wa_params = ['weight_up_precomp_penalty', 'weight_up_postcomp_bonus']
    ed_params = ['elo_denom']
    
    print(f"\n├─ Best Parameters (Full):")
    
    # Print base parameters
    for param_name in base_params:
        if param_name in params:
            value = params[param_name]
            print(f"│  ├─ {param_name}: {repr(value)}")
    
    # Print multiphase decay parameters if present
    has_md = any(p in params for p in md_params)
    if has_md:
        for param_name in md_params:
            if param_name in params:
                value = params[param_name]
                print(f"│  ├─ {param_name}: {repr(value)}")
    
    # Print weight adjustment parameters if present
    has_wa = any(p in params for p in wa_params)
    if has_wa:
        for param_name in wa_params:
            if param_name in params:
                value = params[param_name]
                print(f"│  ├─ {param_name}: {repr(value)}")
    
    # Print elo_denom if present
    if 'elo_denom' in params:
        print(f"│  ├─ elo_denom: {repr(params['elo_denom'])}")
    
    if json_path:
        print(f"│  └─ [Full params saved to: {json_path}]")


def save_master_summary(results, results_dir, seed, generations, population):
    """
    Save master summary JSON with all configurations' best parameters.
    
    Args:
        results: List of result dictionaries from all configurations
        results_dir: Path to results directory
        seed: Random seed used
        generations: Number of generations
        population: Population size
        
    Returns:
        str: Path to the saved master JSON file
    """
    output_file = os.path.join(results_dir, "all_configs_best_params.json")
    
    configs_data = []
    for r in results:
        config_entry = {
            "config_name": r['name'],
            "flags": {
                "multiphase_decay": r.get('multiphase_decay', 'off'),
                "weight_adjust": r.get('weight_adjust', 'off'),
                "optimize_elo_denom": r.get('optimize_elo_denom', 'off')
            },
            "parameters": r.get('best_params', {}),
            "metrics": {
                "train_roi": r.get('train_roi'),
                "oos_roi": r.get('oos_roi'),
                "oos_accuracy": r.get('oos_accuracy'),
                "num_bets": r.get('num_bets'),
                "wins": r.get('wins'),
                "win_rate": r.get('win_rate'),
                "total_wagered": r.get('total_wagered'),
                "total_profit": r.get('total_profit')
            },
            "success": r.get('success', False),
            "elapsed_time": r.get('elapsed_time')
        }
        configs_data.append(config_entry)
    
    master_data = {
        "run_info": {
            "seed": seed,
            "generations": generations,
            "population": population,
            "generated_at": datetime.now().isoformat()
        },
        "configurations": configs_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(master_data, f, indent=2)
    
    return output_file


def print_final_summary(results, results_dir, seed, generations, population):
    """
    Print final summary section after all configurations complete.
    
    Args:
        results: List of result dictionaries from all configurations
        results_dir: Path to results directory
        seed: Random seed used
        generations: Number of generations
        population: Population size
    """
    print("\n" + "=" * 80)
    print("FULL PARAMETER EXPORT SUMMARY")
    print("=" * 80)
    
    print(f"\n📁 Results saved to: {results_dir}/")
    print("\nIndividual configuration files:")
    for r in results:
        if r.get('success') and r.get('best_params'):
            json_file = f"{r['name']}_best_params.json"
            print(f"  ├─ {json_file}")
    
    master_file = "all_configs_best_params.json"
    print(f"  └─ {master_file} (master summary)")
    
    # Print all Python dicts for copy-paste
    print("\n" + "=" * 80)
    print("📋 COPY-PASTE PYTHON DICTIONARIES")
    print("=" * 80)
    
    for r in results:
        if r.get('success') and r.get('best_params'):
            print()
            print(generate_python_dict_output(r['name'], r['best_params']))
    
    # Print usage instructions
    print("\n" + "=" * 80)
    print("USAGE INSTRUCTIONS")
    print("=" * 80)
    print("""
To use the best parameters in main.py:
  1. Copy the Python dictionary above
  2. Import the parameters in your script
  3. Pass them to run_basic_elo_with_mov() or the GA functions

Example:
  from scripts.full_genetic_with_k_denom_mov import run_basic_elo
  
  df = run_basic_elo(df, k=params["k"], mov_params={
      "w_ko": params["w_ko"],
      "w_sub": params["w_sub"],
      "w_udec": params["w_udec"],
      "w_sdec": params["w_sdec"],
      "w_mdec": params["w_mdec"]
  })

To reproduce these results:
  python scripts/run_all_config_tests.py --seed {} --generations {} --population {}
""".format(seed, generations, population))


def run_config(config, script_path, seed, generations, population, timeout=600, 
               config_index=1, total_configs=8, baseline_roi=None, best_config=None,
               results_dir=None):
    """
    Run a single configuration and extract results with real-time progress output.
    
    Args:
        config: dict with 'name' and flag settings
        script_path: Path to the main GA script
        seed: Random seed for reproducibility
        generations: Number of generations
        population: Population size
        timeout: Maximum seconds to wait for subprocess
        config_index: Current configuration number (1-indexed)
        total_configs: Total number of configurations
        baseline_roi: ROI of baseline config for comparison (None if not yet run)
        best_config: Dict with 'name' and 'roi' of current best config
        
    Returns:
        dict: Results including 'oos_roi', 'oos_accuracy', 'best_params', 'success', 'error', 
              'elapsed_time', 'train_roi', 'num_bets', 'wins', 'win_rate', 'total_wagered', 'total_profit'
    """
    result = {
        'name': config['name'],
        'multiphase_decay': config.get('multiphase_decay', 'off'),
        'weight_adjust': config.get('weight_adjust', 'off'),
        'optimize_elo_denom': config.get('optimize_elo_denom', 'off'),
        'oos_roi': None,
        'oos_accuracy': None,
        'best_params': None,
        'train_roi': None,
        'num_bets': None,
        'wins': None,
        'win_rate': None,
        'total_wagered': None,
        'total_profit': None,
        'elapsed_time': None,
        'success': False,
        'error': None
    }
    
    # Build command
    cmd = [
        sys.executable,
        script_path,
        '--mode', 'roi',
        '--generations', str(generations),
        '--population', str(population),
        '--seed', str(seed),
        '--multiphase-decay', config.get('multiphase_decay', 'off'),
        '--weight-adjust', config.get('weight_adjust', 'off'),
        '--optimize-elo-denom', config.get('optimize_elo_denom', 'off'),
    ]
    
    # Format flag display
    md_flag = format_flag(config.get('multiphase_decay', 'off'))
    wa_flag = format_flag(config.get('weight_adjust', 'off'))
    ed_flag = format_flag(config.get('optimize_elo_denom', 'off'))
    
    print(f"\n{'='*70}")
    print(f"[{config_index}/{total_configs}] Running: {config['name']} (md={md_flag} wa={wa_flag} ed={ed_flag})")
    print(f"{'='*70}")
    
    start_time = time.time()
    collected_output = []
    process = None
    
    try:
        # Use Popen for real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            cwd=os.path.dirname(os.path.dirname(script_path))
        )
        
        # Stream output in real-time, filtering for key generation info
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                collected_output.append(line)
                # Show generation progress lines
                if 'Gen ' in line and ('ROI=' in line or 'best=' in line):
                    # This is a generation progress line - show it
                    print(f"  {line.rstrip()}")
                elif '*** NEW BEST' in line:
                    # Highlight new best discoveries
                    print(f"  ⭐ {line.rstrip()}")
                elif 'Initial population' in line:
                    print(f"  {line.rstrip()}")
        
        process.wait(timeout=timeout)
        output = ''.join(collected_output)
        
        if process.returncode != 0:
            result['error'] = f"Process returned code {process.returncode}"
            print(f"  [ERROR] {result['error']}")
        
        # Parse output
        parsed = parse_subprocess_output(output)
        result['oos_roi'] = parsed['oos_roi']
        result['oos_accuracy'] = parsed['oos_accuracy']
        result['best_params'] = parsed.get('best_params')
        result['train_roi'] = parsed.get('train_roi')
        result['num_bets'] = parsed.get('num_bets')
        result['wins'] = parsed.get('wins')
        result['win_rate'] = parsed.get('win_rate')
        result['total_wagered'] = parsed.get('total_wagered')
        result['total_profit'] = parsed.get('total_profit')
        
        if result['oos_roi'] is not None or result['oos_accuracy'] is not None:
            result['success'] = True
        
    except subprocess.TimeoutExpired:
        result['error'] = f"Timeout after {timeout} seconds"
        print(f"  [ERROR] {result['error']}")
        if process is not None:
            process.kill()
    except Exception as e:
        result['error'] = str(e)
        print(f"  [ERROR] {result['error']}")
    
    elapsed = time.time() - start_time
    result['elapsed_time'] = elapsed
    
    # Print enhanced summary for this config
    print(f"\n[{config_index}/{total_configs}] {format_flag('on') if result['success'] else format_flag('off')} {config['name']} (md={md_flag} wa={wa_flag} ed={ed_flag}) [Elapsed: {format_elapsed_time(elapsed)}]")
    
    # Determine if this is the best ROI so far
    is_best = False
    if result['oos_roi'] is not None and best_config is not None:
        if best_config['roi'] is None or result['oos_roi'] > best_config['roi']:
            is_best = True
    elif result['oos_roi'] is not None and best_config is None:
        is_best = True  # First successful config is the best so far
    
    # Train ROI
    if result['train_roi'] is not None:
        print(f"  ├─ Train ROI: {format_roi_with_sign(result['train_roi'])}")
    
    # OOS ROI with comparison
    roi_str = format_roi_with_sign(result['oos_roi'])
    comparison_str = ""
    if baseline_roi is not None and result['oos_roi'] is not None:
        comparison_str = format_roi_comparison(result['oos_roi'], baseline_roi)
    if is_best and result['oos_roi'] is not None:
        print(f"  ├─ OOS ROI: {roi_str}{comparison_str} ⭐")
    else:
        print(f"  ├─ OOS ROI: {roi_str}{comparison_str}")
    
    # OOS Accuracy with bet count
    acc_pct = convert_to_percentage(result['oos_accuracy'])
    if result['num_bets'] is not None and result['wins'] is not None:
        print(f"  ├─ OOS Accuracy: {format_value(acc_pct, '.1f', suffix='%')} ({result['wins']}/{result['num_bets']} bets)")
    else:
        print(f"  ├─ OOS Accuracy: {format_value(acc_pct, '.1f', suffix='%')}")
    
    # Win rate
    if result['win_rate'] is not None:
        print(f"  ├─ Win Rate: {result['win_rate']*100:.1f}%")
    
    # Best parameters - show full precision and save to JSON
    if result['best_params']:
        params = result['best_params']
        
        # Display full parameters with no truncation
        print_full_params_display(config['name'], params)
        
        # Save to JSON if results_dir is provided
        json_path = None
        if results_dir:
            metadata = {
                "seed": seed,
                "generations": generations,
                "population": population,
                "train_roi": result.get('train_roi'),
                "oos_roi": result.get('oos_roi'),
                "oos_accuracy": result.get('oos_accuracy'),
                "num_bets": result.get('num_bets'),
                "wins": result.get('wins'),
                "win_rate": result.get('win_rate'),
                "flags": {
                    "multiphase_decay": config.get('multiphase_decay', 'off'),
                    "weight_adjust": config.get('weight_adjust', 'off'),
                    "optimize_elo_denom": config.get('optimize_elo_denom', 'off')
                }
            }
            json_path = save_params_to_json(config['name'], params, metadata, results_dir)
            print(f"│  └─ [Full params saved to: {json_path}]")
        
        # Print copy-paste Python dict
        print(f"\n📋 Copy-paste into main.py:")
        print(generate_python_dict_output(config['name'], params))
    
    # Total bets and profit
    if result['num_bets'] is not None and result['total_profit'] is not None:
        profit_str = f"${result['total_profit']:.2f}" if result['total_profit'] >= 0 else f"-${abs(result['total_profit']):.2f}"
        print(f"  ├─ Total: {result['num_bets']} bets, {profit_str} profit")
    
    # Status
    status = "COMPLETE ✓" if result['success'] else f"FAILED ✗ ({result['error']})"
    print(f"  └─ Status: {status}")
    
    return result


def generate_all_configs():
    """
    Generate all 8 configuration combinations.
    
    Returns:
        list: List of config dictionaries
    """
    configs = []
    
    for md in ['off', 'on']:
        for wa in ['off', 'on']:
            for ed in ['off', 'on']:
                # Build name showing which flags are on
                name_parts = ['baseline']
                flags = []
                
                if md == 'on':
                    flags.append('md')
                if wa == 'on':
                    flags.append('wa')
                if ed == 'on':
                    flags.append('ed')
                
                if flags:
                    name = '+'.join(flags)
                else:
                    name = 'baseline'
                
                configs.append({
                    'name': name,
                    'multiphase_decay': md,
                    'weight_adjust': wa,
                    'optimize_elo_denom': ed,
                })
    
    return configs


def generate_report(results, seed):
    """
    Generate a comparison report from all results.
    
    Args:
        results: List of result dictionaries
        seed: Random seed used
        
    Returns:
        str: Formatted report
    """
    report_lines = [
        "",
        "=" * 80,
        "CONFIGURATION COMPARISON REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Random Seed: {seed}",
        "",
        "Legend:",
        "  md = multiphase-decay (✓ = ON, ✗ = OFF)",
        "  wa = weight-adjust (✓ = ON, ✗ = OFF)",
        "  ed = optimize-elo-denom (✓ = ON, ✗ = OFF)",
        "",
        "-" * 80,
        f"{'Configuration':<20} {'md':<4} {'wa':<4} {'ed':<4} {'OOS ROI':<12} {'OOS Acc':<10} {'Bets':<6} {'Time':<10} {'Status':<10}",
        "-" * 80,
    ]
    
    # Sort by OOS ROI (descending), putting None values at the end
    sorted_results = sorted(
        results,
        key=lambda x: (x['oos_roi'] is None, -(x['oos_roi'] or float('-inf')))
    )
    
    successful_count = 0
    roi_values = []
    accuracy_values = []
    
    for r in sorted_results:
        # Format flag display with checkmarks
        md_flag = format_flag(r['multiphase_decay'])
        wa_flag = format_flag(r['weight_adjust'])
        ed_flag = format_flag(r['optimize_elo_denom'])
        
        # Format ROI with sign
        roi_str = format_roi_with_sign(r['oos_roi'])
        
        # Format accuracy (convert to percentage if needed)
        acc_pct = convert_to_percentage(r['oos_accuracy'])
        acc_str = format_value(acc_pct, '.1f', suffix='%')
        
        # Format bets count
        bets_str = str(r.get('num_bets', 'N/A')) if r.get('num_bets') is not None else 'N/A'
        
        # Format elapsed time
        time_str = format_elapsed_time(r.get('elapsed_time', 0)) if r.get('elapsed_time') else 'N/A'
        
        # Status
        if r['success']:
            status = "✓ OK"
            successful_count += 1
            if r['oos_roi'] is not None:
                roi_values.append(r['oos_roi'])
            if r['oos_accuracy'] is not None:
                accuracy_values.append(r['oos_accuracy'])
        elif r['error']:
            status = "✗ ERROR"
        else:
            status = "PARTIAL"
        
        report_lines.append(
            f"{r['name']:<20} {md_flag:<4} {wa_flag:<4} {ed_flag:<4} {roi_str:<12} {acc_str:<10} {bets_str:<6} {time_str:<10} {status:<10}"
        )
    
    report_lines.append("-" * 80)
    
    # Add detailed parameters for each configuration - FULL PRECISION
    report_lines.extend([
        "",
        "ALL CONFIGURATION PARAMETERS (FULL PRECISION)",
        "-" * 80,
    ])
    
    # Define parameter groups
    base_params = ['k', 'w_ko', 'w_sub', 'w_udec', 'w_sdec', 'w_mdec']
    md_params = ['quick_succession_days', 'quick_succession_bump', 'decay_days', 'multiphase_decay_rate']
    wa_params = ['weight_up_precomp_penalty', 'weight_up_postcomp_bonus']
    ed_params = ['elo_denom']
    decay_params = ['decay_rate', 'min_days']
    
    for r in sorted_results:
        if r.get('best_params'):
            params = r['best_params']
            report_lines.append(f"\n{r['name']}:")
            
            # Base parameters with full precision
            for param_name in base_params:
                if param_name in params:
                    report_lines.append(f"  {param_name}={repr(params[param_name])}")
            
            # Multiphase decay parameters if present
            has_md = any(p in params for p in md_params)
            if has_md:
                for param_name in md_params:
                    if param_name in params:
                        report_lines.append(f"  {param_name}={repr(params[param_name])}")
            
            # Weight adjustment parameters if present
            has_wa = any(p in params for p in wa_params)
            if has_wa:
                for param_name in wa_params:
                    if param_name in params:
                        report_lines.append(f"  {param_name}={repr(params[param_name])}")
            
            # Elo denom if present
            if 'elo_denom' in params:
                report_lines.append(f"  elo_denom={repr(params['elo_denom'])}")
            
            # Decay params if present
            for param_name in decay_params:
                if param_name in params:
                    report_lines.append(f"  {param_name}={repr(params[param_name])}")
        else:
            report_lines.append(f"\n{r['name']}: Parameters not available")
    
    report_lines.append("-" * 80)
    
    # Summary statistics
    report_lines.extend([
        "",
        "SUMMARY",
        "-" * 40,
        f"Configurations Run: {len(results)}",
        f"Successful: {successful_count}",
        f"Failed: {len(results) - successful_count}",
    ])
    
    if roi_values:
        avg_roi = sum(roi_values) / len(roi_values)
        max_roi = max(roi_values)
        min_roi = min(roi_values)
        best_config = next(r['name'] for r in sorted_results if r['oos_roi'] == max_roi)
        best_result = next(r for r in sorted_results if r['oos_roi'] == max_roi)
        
        report_lines.extend([
            "",
            "ROI Statistics:",
            f"  Best ROI: {format_value(max_roi, '.2f', suffix='%')} ({best_config})",
            f"  Worst ROI: {format_value(min_roi, '.2f', suffix='%')}",
            f"  Average ROI: {format_value(avg_roi, '.2f', suffix='%')}",
        ])
        
        # Add best parameters in a format that can be directly used in main.py
        if best_result.get('best_params'):
            params = best_result['best_params']
            report_lines.extend([
                "",
                "BEST PARAMETERS (copy to main.py):",
                "-" * 40,
                "# Best configuration: " + best_config,
                "df = run_basic_elo_with_mov(df,",
                f"                            k={params.get('k', 32)},",
                f"                            w_ko={params.get('w_ko', 1.4)},",
                f"                            w_sub={params.get('w_sub', 1.3)},",
                f"                            w_udec={params.get('w_udec', 1.0)},",
                f"                            w_sdec={params.get('w_sdec', 0.7)},",
                f"                            w_mdec={params.get('w_mdec', 0.9)})",
            ])
    
    if accuracy_values:
        avg_acc = sum(accuracy_values) / len(accuracy_values)
        max_acc = max(accuracy_values)
        
        # Convert to percentage for display
        avg_acc_pct = convert_to_percentage(avg_acc)
        max_acc_pct = convert_to_percentage(max_acc)
        
        report_lines.extend([
            "",
            "Accuracy Statistics:",
            f"  Best Accuracy: {format_value(max_acc_pct, '.1f', suffix='%')}",
            f"  Average Accuracy: {format_value(avg_acc_pct, '.1f', suffix='%')}",
        ])
    
    # Add reproducibility section
    report_lines.extend([
        "",
        "REPRODUCIBILITY",
        "-" * 40,
        f"To reproduce these results, run:",
        f"  python scripts/run_all_config_tests.py --seed {seed} --generations <N> --population <P>",
        "",
        "To use the best parameters in main.py:",
        "  1. Copy the 'BEST PARAMETERS' code block above",
        "  2. Paste into main.py replacing the existing run_basic_elo_with_mov call",
        "  3. Run main.py to reproduce the Elo calculations",
    ])
    
    report_lines.extend([
        "",
        "=" * 80,
    ])
    
    return '\n'.join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run all configuration tests for genetic algorithm"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--generations', type=int, default=5,
        help='Number of generations for GA (default: 5)'
    )
    parser.add_argument(
        '--population', type=int, default=10,
        help='Population size for GA (default: 10)'
    )
    parser.add_argument(
        '--timeout', type=int, default=600,
        help='Timeout in seconds for each run (default: 600)'
    )
    parser.add_argument(
        '--configs', type=str, default='all',
        help='Which configs to run: "all" or comma-separated list (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Determine script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ga_script = os.path.join(script_dir, 'full_genetic_with_k_denom_mov.py')
    
    if not os.path.exists(ga_script):
        print(f"[ERROR] GA script not found: {ga_script}")
        sys.exit(1)
    
    # Ensure results directory exists
    results_dir = ensure_results_dir(script_dir)
    
    print("=" * 80)
    print("RUN ALL CONFIGURATION TESTS")
    print("=" * 80)
    print(f"Seed: {args.seed} (set for reproducibility)")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Timeout: {args.timeout}s")
    print(f"GA Script: {ga_script}")
    print(f"Results Dir: {results_dir}")
    
    # Generate configurations
    all_configs = generate_all_configs()
    
    # Filter configs if specified
    if args.configs != 'all':
        config_names = [c.strip() for c in args.configs.split(',')]
        all_configs = [c for c in all_configs if c['name'] in config_names]
    
    print(f"\nConfigurations to run: {len(all_configs)}")
    for c in all_configs:
        md_flag = format_flag(c['multiphase_decay'])
        wa_flag = format_flag(c['weight_adjust'])
        ed_flag = format_flag(c['optimize_elo_denom'])
        print(f"  - {c['name']}: md={md_flag} wa={wa_flag} ed={ed_flag}")
    
    # Run all configurations with progress tracking
    results = []
    baseline_roi = None
    best_config = None  # {'name': str, 'roi': float}
    total_start_time = time.time()
    
    for i, config in enumerate(all_configs):
        result = run_config(
            config,
            ga_script,
            args.seed,
            args.generations,
            args.population,
            args.timeout,
            config_index=i + 1,
            total_configs=len(all_configs),
            baseline_roi=baseline_roi,
            best_config=best_config,
            results_dir=results_dir
        )
        results.append(result)
        
        # Track baseline ROI (first successful config with name 'baseline')
        if baseline_roi is None and result['success'] and config['name'] == 'baseline':
            baseline_roi = result['oos_roi']
        
        # Track best configuration
        if result['success'] and result['oos_roi'] is not None:
            if best_config is None or result['oos_roi'] > best_config['roi']:
                best_config = {'name': config['name'], 'roi': result['oos_roi']}
        
        # Show running best after each config
        if best_config is not None:
            print(f"\n🏆 Current Best: {best_config['name']} ({format_roi_with_sign(best_config['roi'])} ROI)")
    
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"All configurations complete! Total time: {format_elapsed_time(total_elapsed)}")
    print(f"{'='*70}")
    
    # Generate and print report
    report = generate_report(results, args.seed)
    print(report)
    
    # Save report to file
    report_path = os.path.join(
        os.path.dirname(script_dir),
        f"config_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not save report: {e}")
    
    # Save master summary JSON with all configurations
    try:
        master_json_path = save_master_summary(
            results, results_dir, args.seed, args.generations, args.population
        )
        print(f"Master summary saved to: {master_json_path}")
    except Exception as e:
        print(f"\n[WARNING] Could not save master summary: {e}")
    
    # Print final summary with all Python dicts and usage instructions
    print_final_summary(results, results_dir, args.seed, args.generations, args.population)
    
    # Return success if at least one config succeeded
    successful = sum(1 for r in results if r['success'])
    if successful == 0:
        print("\n[ERROR] All configurations failed!")
        sys.exit(1)
    elif successful < len(results):
        print(f"\n[WARNING] {len(results) - successful} configuration(s) failed")
        sys.exit(0)
    else:
        print(f"\n[SUCCESS] All {successful} configurations completed successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
