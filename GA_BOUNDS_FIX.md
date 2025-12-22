# GA Bounds Enforcement Fix

## Problem Summary

The genetic algorithm optimization in `optimize_decay_ga_consistency.py` and related scripts was producing:
1. **Exponentially large fitness values** (e.g., 2.667e84)
2. **Negative decay_rate parameters** (e.g., -0.001689, -0.002880)

### Example from Problem Statement:
```
GENERATION 5/15
Best fitness: 2667069243156199678512024101482669793115065885086750167214719652865732205954474046601265944908005376.0000
Best parameters:
  Quick succession days: 117.7
  Quick succession bump: 1.1380x
  Decay days: 316.9
  Decay rate: -0.001689  <-- NEGATIVE!
```

## Root Cause

The issue had two parts:

### 1. Crossover Creates Out-of-Bounds Values
The blend crossover operator (`tools.cxBlend` with `alpha=0.5`) can create children outside the parent range:
- For parents `p1` and `p2`, children can be in range `[min(p1,p2) - 0.5*(max-min), max(p1,p2) + 0.5*(max-min)]`
- When parents are at boundaries (e.g., 0.0001 and 0.01), children can be negative or well outside bounds

Example:
```python
parent1 = [20.0, 1.01, 180.0, 0.0001]  # At lower bounds
parent2 = [120.0, 1.15, 720.0, 0.01]   # At upper bounds
# After crossover:
child1 = [161.4, 1.034, 10.17, -0.0029]  # Out of bounds!
child2 = [-21.4, 1.126, 889.8, 0.0130]   # Out of bounds!
```

### 2. Mutation Creates Negative Values
Gaussian mutation (`tools.mutGaussian` with `sigma=0.2`) adds random noise that can make small positive values negative:
```python
decay_rate = 0.005
# After mutation: decay_rate = -0.285  # NEGATIVE!
```

### 3. Negative Decay Rate Causes Exponential Growth
In `elo/elo_utils.py`, the `apply_multiphase_decay` function uses:
```python
decay_factor = math.exp(-decay_rate * effective_days)
```

When `decay_rate` is negative:
- `-decay_rate * effective_days` becomes positive
- `math.exp(positive_value)` produces exponential **growth** instead of decay
- This causes Elo ratings to explode, leading to massive fitness values

## Solution

Enforce parameter bounds **immediately after** both crossover and mutation operations:

### Before (Buggy Code):
```python
# Apply crossover
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        del child1.fitness.values  # No bounds enforcement!
        del child2.fitness.values

# Apply mutation  
for mutant in offspring:
    if random.random() < MUTPB:
        toolbox.mutate(mutant)
        # Bounds enforced only for mutation
        mutant[0] = max(20, min(120, mutant[0]))
        mutant[1] = max(1.01, min(1.15, mutant[1]))
        mutant[2] = max(180, min(720, mutant[2]))
        mutant[3] = max(0.0001, min(0.01, mutant[3]))
        del mutant.fitness.values
```

### After (Fixed Code):
```python
# Apply crossover
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        # Enforce bounds after crossover
        child1[0] = max(20, min(120, child1[0]))
        child1[1] = max(1.01, min(1.15, child1[1]))
        child1[2] = max(180, min(720, child1[2]))
        child1[3] = max(0.0001, min(0.01, child1[3]))
        child2[0] = max(20, min(120, child2[0]))
        child2[1] = max(1.01, min(1.15, child2[1]))
        child2[2] = max(180, min(720, child2[2]))
        child2[3] = max(0.0001, min(0.01, child2[3]))
        del child1.fitness.values
        del child2.fitness.values

# Apply mutation
for mutant in offspring:
    if random.random() < MUTPB:
        toolbox.mutate(mutant)
        # Enforce bounds after mutation
        mutant[0] = max(20, min(120, mutant[0]))
        mutant[1] = max(1.01, min(1.15, mutant[1]))
        mutant[2] = max(180, min(720, mutant[2]))
        mutant[3] = max(0.0001, min(0.01, mutant[3]))
        del mutant.fitness.values
```

## Files Fixed

1. `optimize_decay_ga_consistency.py` - Main file mentioned in problem statement
2. `optimize_decay_ga.py` - Base GA optimization script
3. `optimize_decay_ga_consistency_v2.py` - Variant
4. `optimize_decay_ga_consistency_working.py` - Variant
5. `optimize_decay_ga_consistency_final.py` - Variant

## Tests Added

### Unit Tests (`tests/test_ga_bounds.py`)
- `test_mutation_can_create_negative_values` - Verifies the bug exists
- `test_mutation_with_bounds_enforcement` - Verifies the fix works
- `test_crossover_can_violate_bounds` - Verifies crossover creates out-of-bounds values
- `test_crossover_with_bounds_enforcement` - Verifies bounds enforcement works
- `test_decay_rate_always_positive` - Ensures decay_rate is always positive

### Integration Test (`tests/test_ga_integration.py`)
- Full GA run with bounds enforcement
- Verifies no exponentially large fitness values
- Verifies all parameters stay within bounds
- Uses the same GA configuration as the main scripts

### Bug Demonstration (`tests/demo_ga_bug.py`)
- Shows what happens WITHOUT bounds enforcement
- Reproduces negative decay_rate values
- Demonstrates the original bug

## Verification

All tests pass:
```bash
$ python3 -m unittest tests.test_ga_bounds -v
test_crossover_can_violate_bounds ... ok
test_crossover_with_bounds_enforcement ... ok
test_decay_rate_always_positive ... ok
test_mutation_can_create_negative_values ... ok
test_mutation_with_bounds_enforcement ... ok

Ran 5 tests in 0.004s

OK

$ python3 tests/test_ga_integration.py
✓ All tests passed! GA bounds enforcement is working correctly.
  - No negative decay_rate values
  - No exponentially large fitness values
  - All parameters stayed within bounds
```

## Impact

After this fix:
- ✅ All decay_rate values will be positive (between 0.0001 and 0.01)
- ✅ All parameters will stay within their specified bounds
- ✅ Fitness values will remain reasonable (no exponential explosion)
- ✅ The GA will converge properly without numerical instability
