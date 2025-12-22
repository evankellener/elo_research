#!/usr/bin/env python3
"""
Quick verification test to ensure fitness values stay reasonable.
Runs just 2 generations with a small population.
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deap import base, creator, tools
import random
from optimize_decay_ga_consistency import (
    evaluate_consistency_fitness, 
    MAX_SHARPE_RATIO,
    MIN_FITNESS
)

# Setup GA
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "Individual"):
    del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("quick_days", random.uniform, 20, 120)
toolbox.register("quick_bump", random.uniform, 1.01, 1.15)
toolbox.register("decay_days", random.uniform, 180, 720)
toolbox.register("decay_rate", random.uniform, 0.0001, 0.01)

toolbox.register("individual", tools.initCycle, creator.Individual,
                (toolbox.quick_days, toolbox.quick_bump, 
                 toolbox.decay_days, toolbox.decay_rate), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_consistency_fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

random.seed(42)

print("="*80)
print("QUICK VERIFICATION TEST - FITNESS BOUNDS")
print("="*80)
print(f"Testing with population=5, generations=2")
print(f"MAX_SHARPE_RATIO = {MAX_SHARPE_RATIO}")
print()

# Small population for quick test
pop = toolbox.population(n=5)

print("Evaluating initial population...")
fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

initial_best = max([ind.fitness.values[0] for ind in pop])
print(f"Initial best fitness: {initial_best:.2f}")

# Verify initial fitness is reasonable
assert initial_best < 100000, f"Initial fitness {initial_best} is too high!"
assert initial_best > MIN_FITNESS, f"Initial fitness {initial_best} is invalid!"
print("✓ Initial fitness is within reasonable bounds (no explosion)\n")

# Run 2 generations
CXPB, MUTPB = 0.7, 0.3
for gen in range(2):
    print(f"Generation {gen + 1}...")
    
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))
    
    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            # Enforce bounds
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
    
    # Mutation
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            # Enforce bounds
            mutant[0] = max(20, min(120, mutant[0]))
            mutant[1] = max(1.01, min(1.15, mutant[1]))
            mutant[2] = max(180, min(720, mutant[2]))
            mutant[3] = max(0.0001, min(0.01, mutant[3]))
            del mutant.fitness.values
    
    # Evaluate
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = list(map(toolbox.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pop[:] = offspring
    
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]
    
    print(f"  Best fitness: {best_fitness:.4f}")
    
    # Verify fitness is reasonable (not in the millions like the bug)
    assert best_fitness < 100000, f"Generation {gen+1} fitness {best_fitness} is too high! BUG NOT FIXED!"
    assert best_fitness > MIN_FITNESS, f"Generation {gen+1} fitness {best_fitness} is invalid!"
    print(f"  ✓ Fitness is within reasonable bounds (no explosion)")

print()
print("="*80)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED")
print("="*80)
print("✓ No fitness explosion detected")
print("✓ All fitness values stayed below 100,000 (vs. 339 million in the bug)")
print("✓ Fix is working correctly!")
