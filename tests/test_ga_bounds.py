"""
Tests for genetic algorithm bounds enforcement.

This module tests that GA operations (crossover and mutation) respect parameter bounds.
"""

import unittest
import random
from deap import base, creator, tools


class TestGABoundsEnforcement(unittest.TestCase):
    """Test bounds enforcement in GA operations."""
    
    def setUp(self):
        """Set up DEAP toolbox for testing."""
        # Clean up any existing creator classes
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.3)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        
        # Define bounds matching optimize_decay_ga_consistency.py
        self.bounds = [
            (20, 120),      # quick_succession_days
            (1.01, 1.15),   # quick_succession_bump
            (180, 720),     # decay_days
            (0.0001, 0.01)  # decay_rate
        ]
        
        random.seed(42)
    
    def tearDown(self):
        """Clean up creator classes."""
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual
    
    def enforce_bounds(self, individual):
        """Enforce bounds on an individual (matches the fix in optimize_decay_ga_consistency.py)."""
        individual[0] = max(20, min(120, individual[0]))
        individual[1] = max(1.01, min(1.15, individual[1]))
        individual[2] = max(180, min(720, individual[2]))
        individual[3] = max(0.0001, min(0.01, individual[3]))
    
    def test_mutation_can_create_negative_values(self):
        """Test that mutation without bounds can create negative decay_rate."""
        found_negative = False
        for i in range(100):
            individual = creator.Individual([50.0, 1.08, 400.0, 0.005])
            self.toolbox.mutate(individual)
            
            if individual[3] < 0:
                found_negative = True
                self.assertLess(individual[3], 0, "Mutation should be able to create negative values")
                break
        
        # This test verifies the problem exists (mutation CAN create negative values)
        self.assertTrue(found_negative, "Mutation should occasionally create negative decay_rate values")
    
    def test_mutation_with_bounds_enforcement(self):
        """Test that bounds enforcement prevents negative values after mutation."""
        for i in range(100):
            individual = creator.Individual([50.0, 1.08, 400.0, 0.005])
            self.toolbox.mutate(individual)
            self.enforce_bounds(individual)
            
            # After bounds enforcement, all values should be within bounds
            for j, (lower, upper) in enumerate(self.bounds):
                self.assertGreaterEqual(individual[j], lower,
                                       f"Parameter {j} below lower bound after mutation+bounds")
                self.assertLessEqual(individual[j], upper,
                                    f"Parameter {j} above upper bound after mutation+bounds")
    
    def test_crossover_can_violate_bounds(self):
        """Test that crossover without bounds can create out-of-bounds values."""
        # Use boundary values as parents
        parent1 = creator.Individual([20.0, 1.01, 180.0, 0.0001])
        parent2 = creator.Individual([120.0, 1.15, 720.0, 0.01])
        
        child1 = self.toolbox.clone(parent1)
        child2 = self.toolbox.clone(parent2)
        self.toolbox.mate(child1, child2)
        
        # Check if any values are out of bounds
        found_violation = False
        for i, (lower, upper) in enumerate(self.bounds):
            if child1[i] < lower or child1[i] > upper:
                found_violation = True
            if child2[i] < lower or child2[i] > upper:
                found_violation = True
        
        # With blend crossover alpha=0.5, we expect violations
        self.assertTrue(found_violation, "Blend crossover should create out-of-bounds values")
    
    def test_crossover_with_bounds_enforcement(self):
        """Test that bounds enforcement keeps crossover children within bounds."""
        # Use boundary values as parents
        parent1 = creator.Individual([20.0, 1.01, 180.0, 0.0001])
        parent2 = creator.Individual([120.0, 1.15, 720.0, 0.01])
        
        child1 = self.toolbox.clone(parent1)
        child2 = self.toolbox.clone(parent2)
        self.toolbox.mate(child1, child2)
        
        # Enforce bounds
        self.enforce_bounds(child1)
        self.enforce_bounds(child2)
        
        # After bounds enforcement, all values should be within bounds
        for i, (lower, upper) in enumerate(self.bounds):
            self.assertGreaterEqual(child1[i], lower,
                                   f"Child1 parameter {i} below lower bound after crossover+bounds")
            self.assertLessEqual(child1[i], upper,
                                f"Child1 parameter {i} above upper bound after crossover+bounds")
            self.assertGreaterEqual(child2[i], lower,
                                   f"Child2 parameter {i} below lower bound after crossover+bounds")
            self.assertLessEqual(child2[i], upper,
                                f"Child2 parameter {i} above upper bound after crossover+bounds")
    
    def test_decay_rate_always_positive(self):
        """Test that decay_rate is always positive after bounds enforcement."""
        # Test many random mutations
        for _ in range(100):
            individual = creator.Individual([50.0, 1.08, 400.0, 0.005])
            self.toolbox.mutate(individual)
            self.enforce_bounds(individual)
            
            self.assertGreater(individual[3], 0,
                             "Decay rate must be positive after bounds enforcement")
        
        # Test crossover with boundary values
        parent1 = creator.Individual([20.0, 1.01, 180.0, 0.0001])
        parent2 = creator.Individual([120.0, 1.15, 720.0, 0.01])
        
        for _ in range(20):
            child1 = self.toolbox.clone(parent1)
            child2 = self.toolbox.clone(parent2)
            self.toolbox.mate(child1, child2)
            
            self.enforce_bounds(child1)
            self.enforce_bounds(child2)
            
            self.assertGreater(child1[3], 0,
                             "Child1 decay rate must be positive after bounds enforcement")
            self.assertGreater(child2[3], 0,
                             "Child2 decay rate must be positive after bounds enforcement")


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
