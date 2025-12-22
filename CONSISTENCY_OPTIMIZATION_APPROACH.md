# Consistency-Focused Optimization Approach

## Objective

Develop a decay parameter optimization strategy that:
1. **Optimizes using ONLY validation data** (last year of training data)
2. **Never touches OOS data** during optimization (true hold-out set)
3. **Prioritizes consistency** over absolute performance
4. **Tests whether consistent validation performance generalizes to OOS**

## Motivation

The previous GA optimization achieved 42.71% OOS ROI but with a large validation-OOS gap:
- Validation ROI: -29.90%
- OOS ROI: +42.71%
- Gap: 72.61%

This large gap raises concerns about:
- **Parameter stability**: Are the results repeatable?
- **Overfitting**: Did we find parameters that work on validation by chance?
- **Generalization**: Will this consistency continue into future events?

## Proposed Solution: Sharpe Ratio Optimization

Instead of optimizing for maximum validation ROI, optimize for **Sharpe ratio** across rolling validation windows:

### Method

1. **Split validation period into quarters** (4 windows)
2. **Calculate ROI for each window independently**
3. **Compute Sharpe ratio**: `mean(window_ROIs) / std(window_ROIs)`
4. **Optimize parameters to maximize Sharpe ratio**

### Fitness Function

```python
Fitness = Sharpe_ratio + 0.1 * max(mean_ROI, 0)
        = (mean_ROI / std_ROI) + bonus_for_positive_ROI
```

### Why This Works

**Sharpe ratio rewards consistency:**
- High mean ROI with low volatility → High Sharpe
- High mean ROI with high volatility → Lower Sharpe  
- Negative ROI → Negative Sharpe (penalized)

**Benefits:**
1. **Finds stable parameters** that work across different time periods
2. **Penalizes volatility** - params must work consistently, not just get lucky once
3. **No data leakage** - OOS is never seen during optimization
4. **Honest assessment** - OOS provides true blind test of generalization

## Implementation

### Script: `optimize_decay_ga_consistency_FINAL.py` (Recommended)

> **Note**: There are multiple versions of the consistency optimization script. See [docs/OPTIMIZE_DECAY_GA_CONSISTENCY_FILES.md](docs/OPTIMIZE_DECAY_GA_CONSISTENCY_FILES.md) for a detailed comparison. The `_FINAL.py` version is the recommended production implementation.

**Key Features:**
- Rolling window analysis on validation data only
- Sharpe ratio as fitness metric
- 4 quarterly windows for stability assessment
- Genetic algorithm with continuous parameter ranges
- Final blind test on OOS data

**Parameter Ranges:**
- Quick succession threshold: 20-120 days
- Quick succession bump: 1.01-1.15x
- Decay threshold: 180-720 days
- Decay rate: 0.0001-0.01

## Expected Outcomes

**If hypothesis is correct:**
- Parameters with high Sharpe on validation will maintain performance on OOS
- Validation-OOS gap will be smaller (more consistent)
- OOS ROI may be lower than 42.71% but more reliable

**If hypothesis is incorrect:**
- Consistency on validation doesn't predict OOS performance
- Gap remains large regardless of optimization approach
- Suggests fundamental distribution shift between validation and OOS periods

## Current Status

**Implementation:** ✓ Complete  
**Testing:** ⚠️ Encountered numerical stability issues

**Issues Found:**
- Negative decay rates slipping through bounds checking
- Infinity fitness values from extreme parameter combinations
- Need additional validation of parameter bounds during mutation

**Next Steps:**
1. Fix bound enforcement in mutation operator
2. Add numerical stability checks (clip extreme probabilities)
3. Re-run optimization with corrected script
4. Compare results to original 42.71% OOS ROI approach

## Alternative Approaches to Consider

If Sharpe ratio optimization doesn't reduce the gap, consider:

1. **Cross-validation on validation set**: Multiple train/test splits within validation
2. **Time-series CV**: Forward-chaining validation to respect temporal order
3. **Ensemble methods**: Average predictions from multiple parameter sets
4. **Regularization**: Add penalty for complex parameters (Occam's razor)
5. **Feature engineering**: Add recency-weighted components to Elo

## Conclusion

The consistency-focused approach provides a principled way to:
- Avoid overfitting to validation data
- Find parameters that generalize reliably
- Maintain experimental rigor (no OOS leakage)
- Test the hypothesis that consistency predicts generalization

This is the correct methodological approach regardless of whether it improves absolute OOS performance.
