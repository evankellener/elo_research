#!/usr/bin/env python3
"""
Integration test to verify GA bounds enforcement prevents negative decay_rate.
This simulates the issue from the problem statement.
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

def test_ga_with_bounds_enforcement():
    """
    Test that GA with proper bounds enforcement doesn't produce negative decay_rate
    or exponentially large fitness values.
    """
    print("Testing GA bounds enforcement integration...")
    
    # Set up DEAP (similar to optimize_decay_ga_consistency.py)
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
    
    # Use simple fitness function for testing
    def simple_fitness(individual):
        quick_succession_days, quick_succession_bump, decay_days, decay_rate = individual
        
        # Check that all parameters are within bounds
        assert 20 <= quick_succession_days <= 120, f"quick_succession_days out of bounds: {quick_succession_days}"
        assert 1.01 <= quick_succession_bump <= 1.15, f"quick_succession_bump out of bounds: {quick_succession_bump}"
        assert 180 <= decay_days <= 720, f"decay_days out of bounds: {decay_days}"
        assert 0.0001 <= decay_rate <= 0.01, f"decay_rate out of bounds: {decay_rate}"
        
        # Test apply_multiphase_decay doesn't cause exponential growth
        elo = 1500
        days_since_fight = 400  # Beyond decay threshold
        
        adjusted_elo = apply_multiphase_decay(
            elo, days_since_fight, quick_succession_days,
            quick_succession_bump, decay_days, decay_rate
        )
        
        # With positive decay_rate, adjusted_elo should be <= original elo
        # (unless in quick succession phase which we're not testing here)
        assert adjusted_elo <= elo, f"Decay caused growth instead of decay: {adjusted_elo} > {elo}"
        
        # Simple fitness that favors lower decay
        return (1.0 / (decay_rate + 0.0001),)
    
    toolbox.register("evaluate", simple_fitness)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # GA parameters
    POPULATION_SIZE = 10
    GENERATIONS = 3
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
    print(f"Initial best fitness: {initial_best:.4f}")
    
    # Verify initial population is within bounds
    for ind in pop:
        assert 20 <= ind[0] <= 120, f"Initial individual param 0 out of bounds: {ind[0]}"
        assert 1.01 <= ind[1] <= 1.15, f"Initial individual param 1 out of bounds: {ind[1]}"
        assert 180 <= ind[2] <= 720, f"Initial individual param 2 out of bounds: {ind[2]}"
        assert 0.0001 <= ind[3] <= 0.01, f"Initial individual param 3 out of bounds: {ind[3]}"
    
    # Run GA
    for gen in range(GENERATIONS):
        print(f"\nGeneration {gen + 1}/{GENERATIONS}")
        
        # Select next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        # Apply crossover WITH bounds enforcement
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
        
        # Apply mutation WITH bounds enforcement
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Enforce bounds after mutation
                mutant[0] = max(20, min(120, mutant[0]))
                mutant[1] = max(1.01, min(1.15, mutant[1]))
                mutant[2] = max(180, min(720, mutant[2]))
                mutant[3] = max(0.0001, min(0.01, mutant[3]))
                del mutant.fitness.values
        
        # Verify all offspring are within bounds
        for ind in offspring:
            assert 20 <= ind[0] <= 120, f"Offspring param 0 out of bounds: {ind[0]}"
            assert 1.01 <= ind[1] <= 1.15, f"Offspring param 1 out of bounds: {ind[1]}"
            assert 180 <= ind[2] <= 720, f"Offspring param 2 out of bounds: {ind[2]}"
            assert 0.0001 <= ind[3] <= 0.01, f"Offspring param 3 out of bounds: {ind[3]}"
        
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
        
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best parameters: {best_ind}")
        
        # Verify fitness is reasonable (not exponentially large)
        assert best_fitness < 1e10, f"Fitness is unreasonably large: {best_fitness}"
        assert not np.isnan(best_fitness), "Fitness is NaN"
        assert not np.isinf(best_fitness), "Fitness is infinite"
    
    print("\n✓ All tests passed! GA bounds enforcement is working correctly.")
    print("  - No negative decay_rate values")
    print("  - No exponentially large fitness values")
    print("  - All parameters stayed within bounds")

if __name__ == "__main__":
    test_ga_with_bounds_enforcement()
