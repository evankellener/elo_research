#!/usr/bin/env python3
"""
Demonstration of the bug: what happens WITHOUT proper bounds enforcement.
This shows the exponentially large fitness values from the problem statement.
"""

import sys
import os
import random
import pandas as pd
import numpy as np
from deap import base, creator, tools

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from elo.elo_utils import apply_multiphase_decay

def test_ga_without_bounds_enforcement():
    """
    Demonstrate that GA WITHOUT proper bounds enforcement produces:
    1. Negative decay_rate values
    2. Exponentially large fitness values
    """
    print("Demonstrating GA bug (without bounds enforcement)...")
    print("This reproduces the issue from the problem statement.\n")
    
    # Set up DEAP
    if hasattr(creator, "FitnessMax"):
        del creator.FitnessMax
    if hasattr(creator, "Individual"):
        del creator.Individual
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Parameter ranges
    toolbox.register("quick_days", random.uniform, 20, 120)
    toolbox.register("quick_bump", random.uniform, 1.01, 1.15)
    toolbox.register("decay_days", random.uniform, 180, 720)
    toolbox.register("decay_rate", random.uniform, 0.0001, 0.01)
    
    toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.quick_days, toolbox.quick_bump, 
                     toolbox.decay_days, toolbox.decay_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Fitness function that shows exponential growth with negative decay_rate
    def fitness_with_negative_decay(individual):
        quick_succession_days, quick_succession_bump, decay_days, decay_rate = individual
        
        # Test apply_multiphase_decay with potentially negative decay_rate
        elo = 1500
        days_since_fight = 400  # Beyond decay threshold
        
        try:
            adjusted_elo = apply_multiphase_decay(
                elo, days_since_fight, quick_succession_days,
                quick_succession_bump, decay_days, decay_rate
            )
            
            # With negative decay_rate, this causes exponential GROWTH
            # fitness based on adjusted_elo will be huge
            fitness = adjusted_elo / elo
            
            return (fitness,)
        except Exception as e:
            return (0.0,)
    
    toolbox.register("evaluate", fitness_with_negative_decay)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # GA parameters
    POPULATION_SIZE = 10
    GENERATIONS = 5
    CXPB = 0.7
    MUTPB = 0.3
    
    random.seed(42)
    
    # Initialize population
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Evaluate initial population
    print("Evaluating initial population...")
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    initial_best = max([ind.fitness.values[0] for ind in pop])
    print(f"Initial best fitness: {initial_best:.4f}\n")
    
    # Run GA WITHOUT proper bounds enforcement
    for gen in range(GENERATIONS):
        print(f"{'='*60}")
        print(f"GENERATION {gen + 1}/{GENERATIONS}")
        print(f"{'='*60}")
        
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover WITHOUT bounds enforcement (OLD BUGGY CODE)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply mutation WITHOUT proper bounds enforcement (OLD BUGGY CODE)
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Only enforce bounds AFTER mutation (too late!)
                # Crossover children are NOT having bounds enforced
                mutant[0] = max(20, min(120, mutant[0]))
                mutant[1] = max(1.01, min(1.15, mutant[1]))
                mutant[2] = max(180, min(720, mutant[2]))
                mutant[3] = max(0.0001, min(0.01, mutant[3]))
                del mutant.fitness.values
        
        # Check for negative decay_rate in offspring (before fitness evaluation)
        negative_count = 0
        for ind in offspring:
            if ind[3] < 0:
                negative_count += 1
        
        if negative_count > 0:
            print(f"\n⚠ WARNING: Found {negative_count} individuals with NEGATIVE decay_rate!")
            for ind in offspring:
                if ind[3] < 0:
                    print(f"  decay_rate = {ind[3]:.6f}")
        
        # Evaluate individuals with invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population
        pop[:] = offspring
        
        # Get best individual of this generation
        best_ind = tools.selBest(pop, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        
        print(f"\nBest fitness: {best_fitness:.4f}")
        print(f"Best parameters:")
        print(f"  Quick succession days: {best_ind[0]:.1f}")
        print(f"  Quick succession bump: {best_ind[1]:.4f}x")
        print(f"  Decay days: {best_ind[2]:.1f}")
        print(f"  Decay rate: {best_ind[3]:.6f}")
        
        if best_fitness > 1e6:
            print(f"\n⚠ EXPONENTIALLY LARGE FITNESS DETECTED: {best_fitness:.2e}")
            print("This matches the issue from the problem statement!")
            break
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Without proper bounds enforcement, the GA produces:")
    print("  ✗ Negative decay_rate values")
    print("  ✗ Exponentially large fitness values")
    print("  ✗ Parameters outside their intended bounds")

if __name__ == "__main__":
    test_ga_without_bounds_enforcement()
