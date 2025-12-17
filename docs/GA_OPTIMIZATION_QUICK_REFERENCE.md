# GA Optimization Quick Reference

Quick reference card for Genetic Algorithm optimization commands.

For the complete guide, see [GA_OPTIMIZATION_GUIDE.md](GA_OPTIMIZATION_GUIDE.md)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run basic example (5 min, learn the system)
python examples/example_ga_optimization.py

# 3. Run ROI optimization (30 min, betting focus)
python optimization/ga_time_split_roi.py

# 4. Run full optimization (60 min, all parameters)
python optimization/full_genetic_with_k_denom_mov.py
```

---

## Main Commands

### Full Multi-Parameter Optimization
```bash
# Optimizes: K, denominator, confidence threshold, 5 MOV weights
# Fitness: Composite (accuracy + log loss + Brier + ROI)
python optimization/full_genetic_with_k_denom_mov.py
```
- **Runtime**: 30-60 minutes
- **Output**: `ga_optimization_history.csv`, plots in `images/`
- **Use for**: Best overall parameters balancing all metrics

---

### ROI-Focused Optimization
```bash
# Default: 6-month time windows
python optimization/ga_time_split_roi.py

# Customize settings
python optimization/ga_time_split_roi.py \
  --data-file data/interleaved_cleaned.csv \
  --split-months 6 \
  --population-size 40 \
  --generations 80 \
  --output-prefix ga_roi
```

**Arguments**:
- `--data-file`: Input CSV (default: `data/interleaved_cleaned.csv`)
- `--split-months`: Validation window size (default: 6)
- `--population-size`: GA population (default: 40)
- `--generations`: Number of generations (default: 80)
- `--output-prefix`: Output file prefix (default: `ga_roi`)

**Runtime**: 20-40 minutes

**Output**: `{prefix}_history.csv`, `{prefix}_best_params.txt`, `images/{prefix}_results.png`

**Use for**: Betting profitability with time-series validation

---

### Basic Example
```bash
# Simple K + denominator optimization (learning)
python examples/example_ga_optimization.py
```
- **Runtime**: ~5 minutes
- **Output**: Console only
- **Use for**: Learning how GA works

---

### Composite Fitness Examples
```bash
# Different fitness weight configurations
python examples/example_composite_fitness.py
```
- **Runtime**: ~5 minutes per example
- **Output**: Console only
- **Use for**: Understanding fitness weight customization

---

## Parameter Ranges

| Parameter | Range | Description |
|-----------|-------|-------------|
| `k` | 10-500 | K-factor (rating change magnitude) |
| `denominator` | 200-600 | Sensitivity parameter (standard: 400) |
| `confidence_threshold` | 30-150 | Minimum Elo diff to bet |
| `w_ko` | 1.0-2.0 | KO/TKO weight |
| `w_sub` | 1.0-2.0 | Submission weight |
| `w_udec` | 0.8-1.5 | Unanimous Decision weight |
| `w_mdec` | 0.7-1.3 | Majority Decision weight |
| `w_sdec` | 0.5-1.2 | Split Decision weight |

---

## Fitness Weights

### Default (Balanced)
```python
fitness_weights = {
    'accuracy': 0.3,      # 30%
    'log_loss': 0.2,      # 20%
    'brier_score': 0.2,   # 20%
    'roi': 0.3            # 30%
}
```

### Accuracy-Focused
```python
fitness_weights = {
    'accuracy': 0.4,
    'log_loss': 0.3,
    'brier_score': 0.3,
    'roi': 0.0
}
```

### Profit-Focused
```python
fitness_weights = {
    'accuracy': 0.15,
    'log_loss': 0.2,
    'brier_score': 0.15,
    'roi': 0.5
}
```

---

## GA Hyperparameters

| Parameter | Default (Full) | Default (Examples) | Description |
|-----------|----------------|-------------------|-------------|
| `population_size` | 50 | 20 | Individuals per generation |
| `generations` | 100 | 20 | Max generations to evolve |
| `elite_size` | 5 | 3 | Best individuals preserved |
| `mutation_rate` | 0.15 | 0.15 | Probability of mutation |
| `crossover_rate` | 0.8 | 0.8 | Probability of crossover |
| `early_stop_generations` | 15 | 10 | Stop if no improvement |

---

## Output Files

### Full Optimization
- `ga_optimization_history.csv` - Full optimization history
- `images/ga_convergence.png` - Fitness over generations
- `images/ga_parameter_evolution.png` - Parameter changes

### ROI Optimization
- `{prefix}_history.csv` - Optimization history
- `{prefix}_best_params.txt` - Best parameters
- `images/{prefix}_results.png` - ROI results plots

---

## Common Use Cases

### 1. General-Purpose Optimization
```bash
python optimization/full_genetic_with_k_denom_mov.py
```
→ Best overall parameters

### 2. Betting Strategy
```bash
python optimization/ga_time_split_roi.py --split-months 6
```
→ Maximize ROI with time-series validation

### 3. Quick Test
```bash
python examples/example_ga_optimization.py
```
→ Learn the system quickly

### 4. Custom Fitness
Edit `full_genetic_with_k_denom_mov.py`:
```python
fitness_weights = {'accuracy': 0.5, 'log_loss': 0.25, 'brier_score': 0.25, 'roi': 0.0}
```
→ Optimize for your specific goals

---

## Interpreting Results

### ROI Values
- **0.05-0.15** (5-15%): Good, sustainable returns
- **0.15-0.30** (15-30%): Excellent returns
- **>0.30** (>30%): Exceptional (verify for overfitting)
- **Negative**: Strategy needs refinement

### Convergence
- ✅ Best fitness increasing → good optimization
- ✅ Average converging to best → population converging
- ⚠️ Flat for 10+ gens → converged or stuck
- ⚠️ Parameters at boundaries → widen search range

---

## Troubleshooting

### Slow Optimization
```python
# Reduce data size
df = df.tail(5000)

# Reduce GA settings
population_size=20
generations=30
```

### Not Improving
```python
# Increase exploration
mutation_rate=0.2
population_size=100

# Check parameter bounds
# Expand if optimal is at boundary
```

### Different Results Each Run
```python
# Set random seed
ga = GeneticAlgorithm(..., random_seed=42)
```

---

## Tips

1. **Start simple**: Run examples first
2. **Validate**: Test on held-out data
3. **Re-optimize**: Every 6-12 months as MMA meta changes
4. **Compare**: Against baselines (default Elo, grid search)
5. **Save results**: Keep logs of optimization runs
6. **Monitor overfitting**: Training vs validation performance

---

## Data Requirements

CSV file with columns:
- `FIGHTER`: Fighter 1 name
- `opp_FIGHTER`: Fighter 2 name
- `result`: 1 = Fighter 1 wins, 0 = Fighter 2 wins
- `DATE`: Fight date (YYYY-MM-DD)
- `METHOD` (optional): For MOV (e.g., "KO", "Submission")

---

## Next Steps

1. Read the [full guide](GA_OPTIMIZATION_GUIDE.md) for detailed explanations
2. Run `python examples/example_ga_optimization.py` to learn
3. Optimize with your data: `python optimization/ga_time_split_roi.py`
4. Apply optimized parameters to your Elo calculations
5. Validate on held-out test data

---

**Full Documentation**: [GA_OPTIMIZATION_GUIDE.md](GA_OPTIMIZATION_GUIDE.md)

**Repository**: [github.com/evankellener/elo_research](https://github.com/evankellener/elo_research)
