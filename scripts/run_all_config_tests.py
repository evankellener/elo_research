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

Usage:
    python scripts/run_all_config_tests.py
    python scripts/run_all_config_tests.py --seed 42
    python scripts/run_all_config_tests.py --generations 5 --population 10
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime


def parse_subprocess_output(output):
    """
    Parse subprocess output to extract ROI and accuracy values.
    
    Args:
        output: String output from subprocess
        
    Returns:
        dict: Contains 'oos_roi' and 'oos_accuracy' (may be None if not found)
    """
    result = {
        'oos_roi': None,
        'oos_accuracy': None,
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


def run_config(config, script_path, seed, generations, population, timeout=600):
    """
    Run a single configuration and extract results.
    
    Args:
        config: dict with 'name' and flag settings
        script_path: Path to the main GA script
        seed: Random seed for reproducibility
        generations: Number of generations
        population: Population size
        timeout: Maximum seconds to wait for subprocess
        
    Returns:
        dict: Results including 'oos_roi', 'oos_accuracy', 'success', 'error'
    """
    result = {
        'name': config['name'],
        'multiphase_decay': config.get('multiphase_decay', 'off'),
        'weight_adjust': config.get('weight_adjust', 'off'),
        'optimize_elo_denom': config.get('optimize_elo_denom', 'off'),
        'oos_roi': None,
        'oos_accuracy': None,
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
    
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.dirname(script_path))
        )
        
        output = process.stdout + process.stderr
        
        if process.returncode != 0:
            result['error'] = f"Process returned code {process.returncode}"
            print(f"  [ERROR] {result['error']}")
            # Still try to parse output even on error
        
        # Parse output
        parsed = parse_subprocess_output(output)
        result['oos_roi'] = parsed['oos_roi']
        result['oos_accuracy'] = parsed['oos_accuracy']
        
        if result['oos_roi'] is not None or result['oos_accuracy'] is not None:
            result['success'] = True
        
    except subprocess.TimeoutExpired:
        result['error'] = f"Timeout after {timeout} seconds"
        print(f"  [ERROR] {result['error']}")
    except Exception as e:
        result['error'] = str(e)
        print(f"  [ERROR] {result['error']}")
    
    # Print summary for this config
    print(f"\nResults for {config['name']}:")
    print(f"  OOS ROI: {format_value(result['oos_roi'], '.2f', suffix='%')}")
    print(f"  OOS Accuracy: {format_value(result['oos_accuracy'], '.1f', suffix='%') if result['oos_accuracy'] is not None and result['oos_accuracy'] <= 1 else format_value(result['oos_accuracy'] * 100 if result['oos_accuracy'] is not None else None, '.1f', suffix='%')}")
    print(f"  Success: {result['success']}")
    
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
        "  md = multiphase-decay",
        "  wa = weight-adjust",
        "  ed = optimize-elo-denom",
        "",
        "-" * 80,
        f"{'Configuration':<20} {'md':<5} {'wa':<5} {'ed':<5} {'OOS ROI':<12} {'OOS Acc':<12} {'Status':<10}",
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
        # Format flag display - use 'o' for on and '-' for off
        md_flag = 'o' if r['multiphase_decay'] == 'on' else '-'
        wa_flag = 'o' if r['weight_adjust'] == 'on' else '-'
        ed_flag = 'o' if r['optimize_elo_denom'] == 'on' else '-'
        
        # Format ROI
        roi_str = format_value(r['oos_roi'], '.2f', suffix='%')
        
        # Format accuracy (convert to percentage if needed)
        if r['oos_accuracy'] is not None:
            acc_value = r['oos_accuracy'] * 100 if r['oos_accuracy'] <= 1 else r['oos_accuracy']
            acc_str = format_value(acc_value, '.1f', suffix='%')
        else:
            acc_str = "N/A"
        
        # Status
        if r['success']:
            status = "OK"
            successful_count += 1
            if r['oos_roi'] is not None:
                roi_values.append(r['oos_roi'])
            if r['oos_accuracy'] is not None:
                accuracy_values.append(r['oos_accuracy'])
        elif r['error']:
            status = "ERROR"
        else:
            status = "PARTIAL"
        
        report_lines.append(
            f"{r['name']:<20} {md_flag:<5} {wa_flag:<5} {ed_flag:<5} {roi_str:<12} {acc_str:<12} {status:<10}"
        )
    
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
        
        report_lines.extend([
            "",
            "ROI Statistics:",
            f"  Best ROI: {format_value(max_roi, '.2f', suffix='%')} ({best_config})",
            f"  Worst ROI: {format_value(min_roi, '.2f', suffix='%')}",
            f"  Average ROI: {format_value(avg_roi, '.2f', suffix='%')}",
        ])
    
    if accuracy_values:
        avg_acc = sum(accuracy_values) / len(accuracy_values)
        max_acc = max(accuracy_values)
        
        # Convert to percentage for display
        avg_acc_pct = avg_acc * 100 if avg_acc <= 1 else avg_acc
        max_acc_pct = max_acc * 100 if max_acc <= 1 else max_acc
        
        report_lines.extend([
            "",
            "Accuracy Statistics:",
            f"  Best Accuracy: {format_value(max_acc_pct, '.1f', suffix='%')}",
            f"  Average Accuracy: {format_value(avg_acc_pct, '.1f', suffix='%')}",
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
    
    # Determine script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ga_script = os.path.join(script_dir, 'full_genetic_with_k_denom_mov.py')
    
    if not os.path.exists(ga_script):
        print(f"[ERROR] GA script not found: {ga_script}")
        sys.exit(1)
    
    print("=" * 80)
    print("RUN ALL CONFIGURATION TESTS")
    print("=" * 80)
    print(f"Seed: {args.seed}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population}")
    print(f"Timeout: {args.timeout}s")
    print(f"GA Script: {ga_script}")
    
    # Generate configurations
    all_configs = generate_all_configs()
    
    # Filter configs if specified
    if args.configs != 'all':
        config_names = [c.strip() for c in args.configs.split(',')]
        all_configs = [c for c in all_configs if c['name'] in config_names]
    
    print(f"\nConfigurations to run: {len(all_configs)}")
    for c in all_configs:
        print(f"  - {c['name']}: md={c['multiphase_decay']}, wa={c['weight_adjust']}, ed={c['optimize_elo_denom']}")
    
    # Run all configurations
    results = []
    for i, config in enumerate(all_configs):
        print(f"\n[{i+1}/{len(all_configs)}] Starting configuration: {config['name']}")
        result = run_config(
            config,
            ga_script,
            args.seed,
            args.generations,
            args.population,
            args.timeout
        )
        results.append(result)
    
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
