import subprocess
import json
import pandas as pd
import itertools
from datetime import datetime
import os
import sys
import time

def run_config_test(config_params):
    """
    Run the GA optimization with a specific configuration of feature flags.    
    
    Args:
        config_params: dict with keys: idx, total, multiphase_decay, weight_adjust, optimize_elo_denom
    
    Returns:
        dict: Results including ROI, accuracy, params, and metrics
    """
    idx = config_params['idx']
    total = config_params['total']
    multiphase_decay = config_params['multiphase_decay']
    weight_adjust = config_params['weight_adjust']
    optimize_elo_denom = config_params['optimize_elo_denom']
    generations = config_params. get('generations', 20)
    base_seed = config_params.get('seed', 42)
    
    # Create unique seed for each config: base_seed + config_index
    seed = base_seed + idx
    
    # Convert booleans to CLI arguments
    multiphase_flag = "on" if multiphase_decay else "off"
    weight_flag = "on" if weight_adjust else "off"
    denom_flag = "on" if optimize_elo_denom else "off"
    
    # Build command
    cmd = [
        "python", "scripts/full_genetic_with_k_denom_mov.py",
        "--mode", "roi",
        "--population", "20",
        "--multiphase-decay", multiphase_flag,
        "--weight-adjust", weight_flag,
        "--optimize-elo-denom", denom_flag,
        "--generations", str(generations),
        "--lookback-days", "270",
        "--seed", str(seed),
    ]
    
    print(f"[{idx}/{total}] Config: {multiphase_flag[:1]}-{weight_flag[:1]}-{denom_flag[:1]} (seed={seed}) | ", end="", flush=True)
    
    process = None
    start_time = time.time()
    
    try:
        # Run the command and capture output in real-time
        process = subprocess. Popen(
            cmd, 
            stdout=subprocess. PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        
        # Read output line by line in real-time
        for line in process.stdout:
            line = line.rstrip('\n')
            output_lines. append(line)
        
        # Wait for process to complete
        return_code = process.wait(timeout=1200)  # 20 minute timeout
        
        output = '\n'.join(output_lines)
        
        # Extract OOS ROI from output
        oos_roi = None
        oos_accuracy = None
        train_roi = None
        
        for line in output.split('\n'):
            if "Best ROI on fights in last" in line or "Best ROI on all fights" in line:
                parts = line.split(':')
                if len(parts) > 1:
                    roi_str = parts[-1].strip(). rstrip('%')
                    try:
                        train_roi = float(roi_str)
                    except ValueError:
                        pass
            elif line.strip(). startswith("ROI:"):
                try:
                    roi_str = line.split(':')[1].strip().rstrip('%')
                    oos_roi = float(roi_str)
                except (ValueError, IndexError):
                    pass
            elif "Accuracy:" in line and "OOS Results:" in output:
                try:
                    acc_str = line.split(':')[1].strip().rstrip('%')
                    oos_accuracy = float(acc_str)
                except (ValueError, IndexError):
                    pass
        
        elapsed = time.time() - start_time
        
        if oos_roi is not None and oos_accuracy is not None:
            print(f"✓ ROI:{oos_roi:+.2f}% Acc:{oos_accuracy:. 1f}% ({elapsed:.0f}s)")
        else:
            print(f"⚠ Partial data ROI:{oos_roi} Acc:{oos_accuracy} ({elapsed:.0f}s)")
        
        return {
            "status": "success",
            "multiphase_decay": multiphase_decay,
            "weight_adjust": weight_adjust,
            "optimize_elo_denom": optimize_elo_denom,
            "train_roi": train_roi,
            "oos_roi": oos_roi,
            "oos_accuracy": oos_accuracy,
            "full_output": output,
            "error": None,
            "elapsed_seconds": elapsed,
            "seed": seed
        }
    
    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT ({elapsed:.0f}s)")
        return {
            "status": "timeout",
            "multiphase_decay": multiphase_decay,
            "weight_adjust": weight_adjust,
            "optimize_elo_denom": optimize_elo_denom,
            "train_roi": None,
            "oos_roi": None,
            "oos_accuracy": None,
            "full_output": None,
            "error": f"Command timed out after {elapsed:.0f}s",
            "elapsed_seconds": elapsed,
            "seed": seed
        }
    
    except Exception as e:
        if process:
            try:
                process.kill()
            except:
                pass
        elapsed = time.time() - start_time
        print(f"✗ ERROR: {str(e)[:50]} ({elapsed:.0f}s)")
        return {
            "status": "error",
            "multiphase_decay": multiphase_decay,
            "weight_adjust": weight_adjust,
            "optimize_elo_denom": optimize_elo_denom,
            "train_roi": None,
            "oos_roi": None,
            "oos_accuracy": None,
            "full_output": None,
            "error": str(e),
            "elapsed_seconds": elapsed,
            "seed": seed
        }


def main(seed=42, generations=20):
    """
    Run all 8 configurations and generate comparison report.  
    
    Args:
        seed: Base random seed for reproducibility (default 42)
        generations: Number of GA generations per config (default 20)
    """
    
    # Generate all combinations (2^3 = 8)
    configurations = list(itertools.product([False, True], repeat=3))
    
    print("\n" + "="*70)
    print("COMPREHENSIVE FEATURE FLAG CONFIGURATION TEST")
    print("="*70)
    print(f"Testing {len(configurations)} configurations (2^3 = 8 combinations)")
    print(f"Base Seed: {seed} (each config gets seed + idx)")
    print(f"Optimizations: {generations} gens, pop=20, lookback=270 days")
    print(f"Start time: {datetime.now()}\n")
    
    # Prepare config params
    config_params_list = []
    for idx, (multiphase, weight, denom) in enumerate(configurations, 1):
        config_params_list.append({
            'idx': idx,
            'total': len(configurations),
            'multiphase_decay': multiphase,
            'weight_adjust': weight,
            'optimize_elo_denom': denom,
            'generations': generations,
            'seed': seed
        })
    
    total_start = time.time()
    
    print("Running configurations (sequential).. .\n")
    print("Config | Status")
    print("-------|" + "─" * 65)
    
    # Run configs sequentially
    results = []
    for params in config_params_list:
        result = run_config_test(params)
        results.append(result)
    
    # Generate comparison report
    print("\n\n" + "="*70)
    print("CONFIGURATION COMPARISON REPORT")
    print("="*70)
    
    # Create summary table
    comparison_df = pd.DataFrame([
        {
            "Config": f"{'T' if r['multiphase_decay'] else 'F'}-{'T' if r['weight_adjust'] else 'F'}-{'T' if r['optimize_elo_denom'] else 'F'}",
            "Multiphase": "✓" if r['multiphase_decay'] else "✗",
            "Weight": "✓" if r['weight_adjust'] else "✗",
            "Denom": "✓" if r['optimize_elo_denom'] else "✗",
            "Train ROI": f"{r['train_roi']:+.2f}%" if r['train_roi'] is not None else "N/A",
            "OOS ROI": f"{r['oos_roi']:+.2f}%" if r['oos_roi'] is not None else "N/A",
            "OOS Acc": f"{r['oos_accuracy']:.2f}%" if r['oos_accuracy'] is not None else "N/A",
            "Status": "✓" if r['status'] == 'success' else "✗",
            "Time": f"{r. get('elapsed_seconds', 0):.0f}s"
        }
        for r in results
    ])
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Find best configuration by OOS ROI
    valid_results = [r for r in results if r['status'] == 'success' and r['oos_roi'] is not None]
    
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['oos_roi'])
        worst_result = min(valid_results, key=lambda r: r['oos_roi'])
        
        print("\n" + "="*70)
        print("FINDINGS")
        print("="*70)
        
        print(f"\n🏆 BEST Configuration (by OOS ROI):")
        print(f"  Multiphase Decay: {'ON' if best_result['multiphase_decay'] else 'OFF'}")
        print(f"  Weight Adjust: {'ON' if best_result['weight_adjust'] else 'OFF'}")
        print(f"  Optimize Elo Denom: {'ON' if best_result['optimize_elo_denom'] else 'OFF'}")
        print(f"  📊 OOS ROI: {best_result['oos_roi']:+. 2f}%")
        print(f"  📊 OOS Accuracy: {best_result['oos_accuracy']:.2f}%")
        print(f"  📊 Train ROI: {best_result['train_roi']:+.2f}%")
        
        print(f"\n📉 WORST Configuration (by OOS ROI):")
        print(f"  Multiphase Decay: {'ON' if worst_result['multiphase_decay'] else 'OFF'}")
        print(f"  Weight Adjust: {'ON' if worst_result['weight_adjust'] else 'OFF'}")
        print(f"  Optimize Elo Denom: {'ON' if worst_result['optimize_elo_denom'] else 'OFF'}")
        print(f"  📊 OOS ROI: {worst_result['oos_roi']:+.2f}%")
        print(f"  📊 OOS Accuracy: {worst_result['oos_accuracy']:.2f}%")
        print(f"  📊 Train ROI: {worst_result['train_roi']:+.2f}%")
        
        print(f"\n📊 IMPROVEMENT (Best vs Worst):")
        print(f"  ROI Delta: {best_result['oos_roi'] - worst_result['oos_roi']:+.2f}%")
        print(f"  Accuracy Delta: {best_result['oos_accuracy'] - worst_result['oos_accuracy']:+. 2f}%")
    
    # Calculate total time
    total_elapsed = time.time() - total_start
    successful = sum(1 for r in results if r['status'] == 'success')
    
    print(f"\n" + "="*70)
    print(f"⏱️  Total Runtime: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"✓ Successful: {successful}/{len(results)}")
    if len(results) > 0:
        print(f"📊 Average per config: {total_elapsed/len(results):. 0f}s")
    print("="*70)
    
    # Save detailed report to file
    report_filename = f"config_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CONFIGURATION COMPARISON REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Base Seed: {seed} (each config = seed + idx)\n")
        f.write(f"Generations: {generations}\n")
        f.write(f"Total Runtime: {total_elapsed/60:. 1f} minutes\n")
        f.write(f"Configurations: {len(results)}\n")
        f. write(f"Successful: {successful}\n")
        f.write(f"Optimizations: {generations} generations, population=20, lookback=270 days\n\n")
        
        f. write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        if valid_results:
            f.write("="*70 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("="*70 + "\n")
            f.write(f"Multiphase Decay: {'ON' if best_result['multiphase_decay'] else 'OFF'}\n")
            f. write(f"Weight Adjust: {'ON' if best_result['weight_adjust'] else 'OFF'}\n")
            f. write(f"Optimize Elo Denom: {'ON' if best_result['optimize_elo_denom'] else 'OFF'}\n")
            f.write(f"OOS ROI: {best_result['oos_roi']:+.2f}%\n")
            f.write(f"OOS Accuracy: {best_result['oos_accuracy']:.2f}%\n")
            f. write(f"Train ROI: {best_result['train_roi']:+.2f}%\n\n")
    
    print(f"\n✅ Report saved to: {report_filename}")
    print(f"End time: {datetime.now()}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all Elo configuration tests")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility (default: 42)")
    parser.add_argument("--generations", type=int, default=20, help="GA generations per config (default: 20)")
    
    args = parser.parse_args()
    
    main(seed=args.seed, generations=args.generations)