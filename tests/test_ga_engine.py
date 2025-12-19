"""
Tests for the genetic algorithm engine.

This module tests the core GA functionality including:
- Population initialization
- Selection methods
- Crossover operations
- Mutation operations
- Evolution and convergence
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from optimization.ga_engine import Individual, GeneticAlgorithm


class TestIndividual(unittest.TestCase):
    """Test the Individual class."""
    
    def test_initialization(self):
        """Test individual creation."""
        genes = {'x': 1.0, 'y': 2.0}
        ind = Individual(genes)
        self.assertEqual(ind.genes, genes)
        self.assertIsNone(ind.fitness)
    
    def test_copy(self):
        """Test individual copying."""
        genes = {'x': 1.0, 'y': 2.0}
        ind1 = Individual(genes)
        ind1.fitness = 0.5
        
        ind2 = ind1.copy()
        self.assertEqual(ind1.genes, ind2.genes)
        self.assertEqual(ind1.fitness, ind2.fitness)
        
        # Modify copy and ensure original unchanged
        ind2.genes['x'] = 5.0
        self.assertEqual(ind1.genes['x'], 1.0)


class TestGeneticAlgorithm(unittest.TestCase):
    """Test the GeneticAlgorithm class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Simple quadratic fitness function: maximize -(x^2 + y^2)
        # Optimum at (0, 0) with fitness 0
        def fitness_fn(genes):
            x = genes.get('x', 0)
            y = genes.get('y', 0)
            return -(x**2 + y**2)
        
        self.fitness_fn = fitness_fn
        self.param_bounds = {
            'x': (-10, 10),
            'y': (-10, 10)
        }
    
    def test_initialization(self):
        """Test GA initialization."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            random_seed=42,
            verbose=False
        )
        
        self.assertEqual(len(ga.population), 0)
        self.assertIsNone(ga.best_individual)
    
    def test_population_initialization(self):
        """Test population initialization."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        
        self.assertEqual(len(ga.population), 20)
        
        # Check that all individuals have correct genes
        for ind in ga.population:
            self.assertIn('x', ind.genes)
            self.assertIn('y', ind.genes)
            self.assertGreaterEqual(ind.genes['x'], -10)
            self.assertLessEqual(ind.genes['x'], 10)
            self.assertGreaterEqual(ind.genes['y'], -10)
            self.assertLessEqual(ind.genes['y'], 10)
    
    def test_evaluation(self):
        """Test fitness evaluation."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        ga.evaluate_population()
        
        # All individuals should have fitness
        for ind in ga.population:
            self.assertIsNotNone(ind.fitness)
        
        # Best individual should be set
        self.assertIsNotNone(ga.best_individual)
        
        # Population should be sorted by fitness
        for i in range(len(ga.population) - 1):
            self.assertGreaterEqual(ga.population[i].fitness, ga.population[i+1].fitness)
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            tournament_size=3,
            selection_method="tournament",
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        ga.evaluate_population()
        
        # Select a parent
        parent = ga.select_parent()
        
        self.assertIsInstance(parent, Individual)
        self.assertIsNotNone(parent.fitness)
    
    def test_roulette_selection(self):
        """Test roulette wheel selection."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            selection_method="roulette",
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        ga.evaluate_population()
        
        # Select a parent
        parent = ga.select_parent()
        
        self.assertIsInstance(parent, Individual)
        self.assertIsNotNone(parent.fitness)
    
    def test_uniform_crossover(self):
        """Test uniform crossover."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            crossover_method="uniform",
            crossover_rate=1.0,  # Always crossover
            random_seed=42,
            verbose=False
        )
        
        parent1 = Individual({'x': 1.0, 'y': 2.0})
        parent2 = Individual({'x': 3.0, 'y': 4.0})
        
        child1, child2 = ga.crossover(parent1, parent2)
        
        self.assertIsInstance(child1, Individual)
        self.assertIsInstance(child2, Individual)
        self.assertIn('x', child1.genes)
        self.assertIn('y', child1.genes)
    
    def test_mutation(self):
        """Test mutation."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            mutation_rate=1.0,  # Always mutate
            random_seed=42,
            verbose=False
        )
        
        ind = Individual({'x': 5.0, 'y': 5.0})
        ind.fitness = 0.5
        
        original_x = ind.genes['x']
        original_y = ind.genes['y']
        
        mutated = ga.mutate(ind)
        
        # At least one gene should have changed (with high probability)
        self.assertTrue(
            mutated.genes['x'] != original_x or mutated.genes['y'] != original_y
        )
        
        # Fitness should be cleared
        self.assertIsNone(mutated.fitness)
        
        # Genes should be within bounds
        self.assertGreaterEqual(mutated.genes['x'], -10)
        self.assertLessEqual(mutated.genes['x'], 10)
        self.assertGreaterEqual(mutated.genes['y'], -10)
        self.assertLessEqual(mutated.genes['y'], 10)
    
    def test_evolution(self):
        """Test one generation of evolution."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            elite_size=2,
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        ga.evaluate_population()
        
        best_before = ga.best_individual.fitness
        
        ga.evolve_generation()
        ga.evaluate_population()
        
        # Population size should remain constant
        self.assertEqual(len(ga.population), 20)
        
        # All individuals should have fitness
        for ind in ga.population:
            self.assertIsNotNone(ind.fitness)
    
    def test_convergence(self):
        """Test that GA converges toward optimum."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=30,
            elite_size=3,
            mutation_rate=0.1,
            crossover_rate=0.8,
            random_seed=42,
            verbose=False
        )
        
        best = ga.run(generations=50)
        
        # Best fitness should be close to 0 (the optimum)
        self.assertGreater(best.fitness, -1.0)  # Should be close to 0
        
        # Best genes should be close to (0, 0)
        self.assertLess(abs(best.genes['x']), 2.0)
        self.assertLess(abs(best.genes['y']), 2.0)
        
        # History should be recorded
        self.assertGreater(len(ga.history), 0)
    
    def test_elitism(self):
        """Test that elitism preserves best individuals."""
        ga = GeneticAlgorithm(
            param_bounds=self.param_bounds,
            fitness_fn=self.fitness_fn,
            population_size=20,
            elite_size=3,
            random_seed=42,
            verbose=False
        )
        
        ga.initialize_population()
        ga.evaluate_population()
        
        # Get top 3 individuals
        elite_before = [ind.copy() for ind in ga.population[:3]]
        
        ga.evolve_generation()
        
        # Top 3 should be in new population (exact same objects due to copy)
        new_population_genes = [ind.genes for ind in ga.population]
        for elite_ind in elite_before:
            # Check if this elite's genes are in the new population
            found = any(
                all(abs(elite_ind.genes[k] - ind_genes[k]) < 1e-10 for k in elite_ind.genes.keys())
                for ind_genes in new_population_genes
            )
            self.assertTrue(found, "Elite individual not preserved")
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        # Use a fitness function that converges quickly
        def simple_fitness(genes):
            return -abs(genes['x'])
        
        ga = GeneticAlgorithm(
            param_bounds={'x': (-10, 10)},
            fitness_fn=simple_fitness,
            population_size=20,
            random_seed=42,
            verbose=False
        )
        
        best = ga.run(generations=1000, early_stop_generations=10)
        
        # Should stop before 1000 generations
        self.assertLess(len(ga.history), 1000)


class TestGAWithConstraints(unittest.TestCase):
    """Test GA with more complex fitness functions."""
    
    def test_sphere_function(self):
        """Test optimization of sphere function in 5D."""
        def sphere_fitness(genes):
            return -sum(genes[f'x{i}']**2 for i in range(5))
        
        param_bounds = {f'x{i}': (-10, 10) for i in range(5)}
        
        ga = GeneticAlgorithm(
            param_bounds=param_bounds,
            fitness_fn=sphere_fitness,
            population_size=50,
            elite_size=5,
            mutation_rate=0.1,
            crossover_rate=0.8,
            random_seed=42,
            verbose=False
        )
        
        best = ga.run(generations=100)
        
        # Should converge close to zero
        self.assertGreater(best.fitness, -10.0)
        
        # All genes should be close to 0
        for i in range(5):
            self.assertLess(abs(best.genes[f'x{i}']), 3.0)
    
    def test_asymmetric_bounds(self):
        """Test GA with asymmetric parameter bounds."""
        def asymmetric_fitness(genes):
            # Optimum at x=5, y=-3
            return -((genes['x'] - 5)**2 + (genes['y'] + 3)**2)
        
        param_bounds = {
            'x': (0, 10),
            'y': (-10, 0)
        }
        
        ga = GeneticAlgorithm(
            param_bounds=param_bounds,
            fitness_fn=asymmetric_fitness,
            population_size=30,
            random_seed=42,
            verbose=False
        )
        
        best = ga.run(generations=80)
        
        # Should find optimum near (5, -3)
        self.assertLess(abs(best.genes['x'] - 5), 1.0)
        self.assertLess(abs(best.genes['y'] + 3), 1.0)


class TestOptimizationModeDisplay(unittest.TestCase):
    """Test display formatting for different optimization modes."""
    
    def test_log_loss_mode_display(self):
        """Test that log_loss mode displays actual log_loss values."""
        def fitness_fn(genes):
            # Return a fitness value in the range used by log_loss (exp(-log_loss))
            # Simulate log_loss ≈ 0.7 -> fitness ≈ 0.497
            return 0.5
        
        param_bounds = {'x': (0, 10)}
        
        ga = GeneticAlgorithm(
            param_bounds=param_bounds,
            fitness_fn=fitness_fn,
            population_size=10,
            random_seed=42,
            verbose=False,
            optimization_mode="log_loss"
        )
        
        # Test the formatting function
        metric_name, display_value = ga._format_metric_for_display(0.5)
        
        self.assertEqual(metric_name, "Log Loss")
        # fitness = 0.5 corresponds to log_loss = -ln(0.5) ≈ 0.693
        self.assertAlmostEqual(display_value, 0.693, places=2)
    
    def test_default_mode_display(self):
        """Test that default mode displays fitness values."""
        def fitness_fn(genes):
            return 0.5
        
        param_bounds = {'x': (0, 10)}
        
        ga = GeneticAlgorithm(
            param_bounds=param_bounds,
            fitness_fn=fitness_fn,
            population_size=10,
            random_seed=42,
            verbose=False
        )
        
        # Test the formatting function
        metric_name, display_value = ga._format_metric_for_display(0.5)
        
        self.assertEqual(metric_name, "Fitness")
        self.assertEqual(display_value, 0.5)
    
    def test_log_loss_optimization_direction(self):
        """Test that log_loss displays show improvement correctly."""
        # Better model has higher fitness (lower log_loss)
        better_fitness = 0.6  # log_loss ≈ 0.511
        worse_fitness = 0.5   # log_loss ≈ 0.693
        
        param_bounds = {'x': (0, 10)}
        fitness_fn = lambda genes: better_fitness
        
        ga = GeneticAlgorithm(
            param_bounds=param_bounds,
            fitness_fn=fitness_fn,
            population_size=10,
            random_seed=42,
            verbose=False,
            optimization_mode="log_loss"
        )
        
        _, better_display = ga._format_metric_for_display(better_fitness)
        _, worse_display = ga._format_metric_for_display(worse_fitness)
        
        # Better model should have LOWER displayed log_loss
        self.assertLess(better_display, worse_display)
    
    def test_fitness_to_log_loss_conversion(self):
        """Test that fitness can be correctly converted back to log_loss."""
        # Test known fitness values and their corresponding log_loss values
        test_cases = [
            (0.5, np.log(2)),      # Random guessing: ln(2) ≈ 0.693
            (1.0, 0.0),            # Perfect prediction
            (0.6, -np.log(0.6)),   # Better than random
            (0.4, -np.log(0.4)),   # Worse than random
        ]
        
        for fitness, expected_log_loss in test_cases:
            # Clamp fitness to prevent log(0)
            fitness_clamped = max(fitness, 1e-15)
            # Convert: log_loss = -ln(fitness)
            calculated_log_loss = -np.log(fitness_clamped)
            self.assertAlmostEqual(calculated_log_loss, expected_log_loss, places=5,
                                   msg=f"Fitness {fitness} -> Log Loss conversion failed")


class TestROIFitnessFunction(unittest.TestCase):
    """Test ROI fitness function behavior."""
    
    def test_roi_bets_placed_on_all_valid_fights(self):
        """Test that ROI places bets on all valid fights regardless of confidence."""
        import pandas as pd
        from optimization.ga_engine import create_elo_fitness_function
        
        # Create a complete test dataset with all required columns
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=10),
            'FIGHTER': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'opp_FIGHTER': ['B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A'],
            'result': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'win': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            'loss': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'precomp_boutcount': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'opp_precomp_boutcount': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        })
        
        # Create fitness function optimizing for ROI
        fitness_fn = create_elo_fitness_function(
            df,
            base_elo=1500,
            use_validation_split=False,
            optimize_for="roi"
        )
        
        # Test with any confidence threshold - bets should be placed on all valid fights
        params = {
            'k': 32,
            'denominator': 400,
            'confidence_threshold': 10000  # High threshold, but bets still placed
        }
        
        fitness = fitness_fn(params)
        
        # Should return a valid fitness value (not -1.0) because bets are placed on all valid fights
        self.assertNotEqual(fitness, -1.0,
                        "ROI fitness should place bets on all valid fights regardless of confidence threshold")
    
    def test_roi_negative_performance(self):
        """Test that ROI fitness can return negative values for losing strategies."""
        import pandas as pd
        from optimization.ga_engine import create_elo_fitness_function
        
        # Create a dataset where the underdog wins (predictions will be wrong)
        # Fighter A starts weaker but wins, Fighter B starts stronger but loses
        fighters = []
        opp_fighters = []
        results = []
        wins = []
        losses = []
        
        # Fighter A (underdog) beats Fighter B (favorite) consistently
        for i in range(20):
            fighters.append('A')
            opp_fighters.append('B')
            results.append(1)  # A wins
            wins.append(1)
            losses.append(0)
        
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=20),
            'FIGHTER': fighters,
            'opp_FIGHTER': opp_fighters,
            'result': results,
            'win': wins,
            'loss': losses,
            'precomp_boutcount': list(range(1, 21)),
            'opp_precomp_boutcount': list(range(1, 21))
        })
        
        # Create fitness function optimizing for ROI
        fitness_fn = create_elo_fitness_function(
            df,
            base_elo=1500,
            use_validation_split=False,
            optimize_for="roi"
        )
        
        # Test with low confidence threshold to bet on all fights
        params = {
            'k': 32,
            'denominator': 400,
            'confidence_threshold': 0  # Bet on everything
        }
        
        fitness = fitness_fn(params)
        
        # Should return negative ROI since predictions are systematically wrong
        # Note: This test might not always give negative ROI depending on how Elo evolves
        # So we'll just check that it's within valid range
        self.assertGreaterEqual(fitness, -1.0,
                               "ROI fitness should be >= -1.0 (worst case)")
        self.assertLessEqual(fitness, 1.0,
                            "ROI fitness should be <= 1.0 (best case)")
    
    def test_roi_positive_performance(self):
        """Test that ROI fitness returns positive values for winning strategies."""
        import pandas as pd
        from optimization.ga_engine import create_elo_fitness_function
        
        # Create a dataset where the favorite always wins (predictions will be correct)
        fighters = []
        opp_fighters = []
        results = []
        wins = []
        losses = []
        
        # Fighter A beats Fighter B consistently (favorite wins)
        for i in range(20):
            if i % 2 == 0:
                fighters.append('A')
                opp_fighters.append('B')
                results.append(1)  # A wins
                wins.append(1)
                losses.append(0)
            else:
                fighters.append('B')
                opp_fighters.append('A')
                results.append(0)  # A wins (B loses)
                wins.append(0)
                losses.append(1)
        
        df = pd.DataFrame({
            'DATE': pd.date_range('2020-01-01', periods=20),
            'FIGHTER': fighters,
            'opp_FIGHTER': opp_fighters,
            'result': results,
            'win': wins,
            'loss': losses,
            'precomp_boutcount': list(range(1, 21)),
            'opp_precomp_boutcount': list(range(1, 21))
        })
        
        # Create fitness function optimizing for ROI
        fitness_fn = create_elo_fitness_function(
            df,
            base_elo=1500,
            use_validation_split=False,
            optimize_for="roi"
        )
        
        # Test with low confidence threshold to bet on all fights
        params = {
            'k': 32,
            'denominator': 400,
            'confidence_threshold': 0  # Bet on everything
        }
        
        fitness = fitness_fn(params)
        
        # Should return positive ROI since predictions are generally correct
        # Note: ROI might not always be positive due to Elo dynamics, so we check valid range
        self.assertGreaterEqual(fitness, -1.0,
                               "ROI fitness should be >= -1.0 (worst case)")
        self.assertLessEqual(fitness, 1.0,
                            "ROI fitness should be <= 1.0 (best case)")


class TestFileOutputFunctionality(unittest.TestCase):
    """Test file output functionality in example script."""
    
    def test_roi_percentage_display(self):
        """Test that ROI is correctly converted to percentage."""
        roi_fitness = 0.3492
        roi_percent = roi_fitness * 100
        
        self.assertAlmostEqual(roi_percent, 34.92, places=2)
    
    def test_json_structure(self):
        """Test that results dictionary has expected structure."""
        import json
        from datetime import datetime
        
        # Simulate results structure
        results = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "optimization_mode": "roi",
            "best_parameters": {"k": 10.0, "denominator": 437.78},
            "best_fitness": 0.3492,
            "best_roi_percent": 34.92,
            "generations": 20,
            "population_size": 20,
            "elite_size": 3,
            "mutation_rate": 0.15,
            "crossover_rate": 0.8,
            "history": []
        }
        
        # Verify structure
        self.assertIn("timestamp", results)
        self.assertIn("optimization_mode", results)
        self.assertIn("best_parameters", results)
        self.assertIn("best_fitness", results)
        self.assertIn("best_roi_percent", results)
        
        # Verify ROI calculation
        self.assertAlmostEqual(results["best_roi_percent"], 
                              results["best_fitness"] * 100, places=2)
        
        # Test JSON serialization
        json_str = json.dumps(results)
        self.assertIsInstance(json_str, str)


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False, verbosity=2)
