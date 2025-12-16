"""
Genetic Algorithm Engine for Elo Parameter Optimization

This module implements a flexible genetic algorithm framework that can be used
to optimize Elo rating system parameters including:
- K-factor
- Denominator (sensitivity parameter)
- Method of Victory (MOV) weights
- Decay rates and other advanced features

The GA uses standard evolutionary operators:
- Selection: Tournament selection or roulette wheel selection
- Crossover: Single-point, multi-point, or uniform crossover
- Mutation: Gaussian mutation with adaptive rates
- Elitism: Preserve best individuals across generations
"""

import random
import numpy as np
from typing import List, Dict, Callable, Tuple, Optional
import copy


class Individual:
    """Represents a single candidate solution in the genetic algorithm."""
    
    def __init__(self, genes: Dict[str, float]):
        """
        Initialize an individual with a parameter set.
        
        Args:
            genes: Dictionary mapping parameter names to values
        """
        self.genes = genes.copy()
        self.fitness = None
        
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f if self.fitness else 'None'}, genes={self.genes})"
    
    def copy(self):
        """Create a deep copy of this individual."""
        new_ind = Individual(self.genes)
        new_ind.fitness = self.fitness
        return new_ind


class GeneticAlgorithm:
    """
    Genetic Algorithm optimizer for Elo rating system parameters.
    
    This class implements a standard genetic algorithm with support for:
    - Customizable fitness functions
    - Multiple selection strategies
    - Multiple crossover strategies
    - Adaptive mutation
    - Elitism
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        fitness_fn: Callable[[Dict[str, float]], float],
        population_size: int = 50,
        elite_size: int = 5,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        selection_method: str = "tournament",
        crossover_method: str = "uniform",
        random_seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) tuples
            fitness_fn: Function that takes a parameter dict and returns fitness score
            population_size: Number of individuals in the population
            elite_size: Number of best individuals to preserve each generation
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between parents
            tournament_size: Size of tournament for tournament selection
            selection_method: "tournament" or "roulette"
            crossover_method: "single", "two_point", or "uniform"
            random_seed: Random seed for reproducibility
            verbose: Print progress messages
        """
        self.param_bounds = param_bounds
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.verbose = verbose
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []
        
    def initialize_population(self):
        """Create initial population with random parameter values."""
        self.population = []
        for _ in range(self.population_size):
            genes = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                genes[param] = random.uniform(min_val, max_val)
            self.population.append(Individual(genes))
        
        if self.verbose:
            print(f"Initialized population of {self.population_size} individuals")
    
    def evaluate_population(self):
        """Evaluate fitness for all individuals in the population."""
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.fitness_fn(individual.genes)
        
        # Update best individual
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = self.population[0].copy()
    
    def select_parent(self) -> Individual:
        """
        Select a parent using the configured selection method.
        
        Returns:
            Selected individual
        """
        if self.selection_method == "tournament":
            return self._tournament_selection()
        elif self.selection_method == "roulette":
            return self._roulette_selection()
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
    
    def _tournament_selection(self) -> Individual:
        """Select parent using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self) -> Individual:
        """Select parent using roulette wheel selection."""
        # Ensure all fitness values are positive by shifting if needed
        min_fitness = min(ind.fitness for ind in self.population)
        if min_fitness < 0:
            fitness_values = [ind.fitness - min_fitness + 1e-6 for ind in self.population]
        else:
            fitness_values = [ind.fitness for ind in self.population]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(self.population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_values):
            current += fitness
            if current >= pick:
                return self.population[i]
        
        return self.population[-1]
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Create two offspring from two parents using crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring individuals
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.crossover_method == "single":
            return self._single_point_crossover(parent1, parent2)
        elif self.crossover_method == "two_point":
            return self._two_point_crossover(parent1, parent2)
        elif self.crossover_method == "uniform":
            return self._uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Unknown crossover method: {self.crossover_method}")
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        params = list(self.param_bounds.keys())
        if len(params) <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(params) - 1)
        
        child1_genes = {}
        child2_genes = {}
        for i, param in enumerate(params):
            if i < crossover_point:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover."""
        params = list(self.param_bounds.keys())
        if len(params) <= 2:
            return self._single_point_crossover(parent1, parent2)
        
        point1 = random.randint(1, len(params) - 2)
        point2 = random.randint(point1 + 1, len(params) - 1)
        
        child1_genes = {}
        child2_genes = {}
        for i, param in enumerate(params):
            if point1 <= i < point2:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
            else:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover - each gene has 50% chance to come from either parent."""
        child1_genes = {}
        child2_genes = {}
        
        for param in self.param_bounds.keys():
            if random.random() < 0.5:
                child1_genes[param] = parent1.genes[param]
                child2_genes[param] = parent2.genes[param]
            else:
                child1_genes[param] = parent2.genes[param]
                child2_genes[param] = parent1.genes[param]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Apply Gaussian mutation to an individual's genes.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual (modified in place)
        """
        for param, (min_val, max_val) in self.param_bounds.items():
            if random.random() < self.mutation_rate:
                # Gaussian mutation with standard deviation = 10% of range
                std_dev = (max_val - min_val) * 0.1
                mutation = np.random.normal(0, std_dev)
                individual.genes[param] += mutation
                
                # Clamp to bounds
                individual.genes[param] = max(min_val, min(max_val, individual.genes[param]))
        
        # Clear fitness since genes changed
        individual.fitness = None
        return individual
    
    def evolve_generation(self):
        """Evolve the population by one generation."""
        # Keep elite individuals
        new_population = [ind.copy() for ind in self.population[:self.elite_size]]
        
        # Generate offspring to fill the rest of the population
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            child1, child2 = self.crossover(parent1, parent2)
            
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        self.population = new_population
    
    def run(self, generations: int, early_stop_generations: Optional[int] = None) -> Individual:
        """
        Run the genetic algorithm for a specified number of generations.
        
        Args:
            generations: Number of generations to evolve
            early_stop_generations: Stop if no improvement for this many generations
            
        Returns:
            Best individual found
        """
        self.initialize_population()
        self.evaluate_population()
        
        best_fitness_history = []
        generations_without_improvement = 0
        
        for gen in range(generations):
            self.evolve_generation()
            self.evaluate_population()
            
            # Record statistics
            avg_fitness = np.mean([ind.fitness for ind in self.population])
            best_fitness = self.best_individual.fitness
            worst_fitness = self.population[-1].fitness
            
            self.history.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'worst_fitness': worst_fitness,
                'best_genes': self.best_individual.genes.copy()
            })
            
            if self.verbose:
                print(f"Generation {gen + 1}/{generations}: "
                      f"Best={best_fitness:.6f}, Avg={avg_fitness:.6f}, "
                      f"Worst={worst_fitness:.6f}")
            
            # Early stopping check
            if early_stop_generations and len(best_fitness_history) > 0:
                if best_fitness <= best_fitness_history[-1]:
                    generations_without_improvement += 1
                else:
                    generations_without_improvement = 0
                
                if generations_without_improvement >= early_stop_generations:
                    if self.verbose:
                        print(f"\nEarly stopping: No improvement for {early_stop_generations} generations")
                    break
            
            best_fitness_history.append(best_fitness)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Optimization Complete!")
            print(f"{'='*60}")
            print(f"Best fitness: {self.best_individual.fitness:.6f}")
            print(f"Best parameters:")
            for param, value in self.best_individual.genes.items():
                print(f"  {param}: {value:.4f}")
        
        return self.best_individual
    
    def get_history_dataframe(self):
        """
        Convert optimization history to a pandas DataFrame.
        
        Returns:
            DataFrame with columns: generation, best_fitness, avg_fitness, worst_fitness
        """
        import pandas as pd
        return pd.DataFrame(self.history)


def create_elo_fitness_function(
    df,
    test_df=None,
    base_elo: float = 1500,
    use_validation_split: bool = True,
    validation_percentile: float = 0.8,
    optimize_for: str = "accuracy"
) -> Callable[[Dict[str, float]], float]:
    """
    Create a fitness function for Elo parameter optimization.
    
    This function returns a fitness function that can be used with the genetic
    algorithm to optimize Elo parameters.
    
    Args:
        df: Training DataFrame with fight data
        test_df: Optional test DataFrame for out-of-sample evaluation
        base_elo: Base Elo rating for new fighters
        use_validation_split: Whether to use time-based validation split
        validation_percentile: Percentile for validation split
        optimize_for: "accuracy" for prediction accuracy or "roi" for betting ROI
        
    Returns:
        Fitness function that takes parameter dict and returns fitness score
    """
    from optimization.optimal_k_with_mov import run_basic_elo, elo_accuracy
    
    if use_validation_split:
        cutoff = df["DATE"].quantile(validation_percentile)
    else:
        cutoff = None
    
    def fitness_function(params: Dict[str, float]) -> float:
        """Evaluate fitness for a parameter set."""
        # Extract parameters with defaults
        k = params.get("k", 32)
        denominator = params.get("denominator", 400)
        
        # MOV weights (use defaults if not in params)
        use_mov = any(key.startswith("w_") for key in params.keys())
        
        # Run Elo with these parameters
        if use_mov:
            # Create a modified version of run_basic_elo that uses custom MOV weights
            # For now, use the standard run_basic_elo
            trial = run_basic_elo(
                df.copy(),
                k=k,
                base_elo=base_elo,
                denominator=denominator,
                use_mov=True
            )
        else:
            trial = run_basic_elo(
                df.copy(),
                k=k,
                base_elo=base_elo,
                denominator=denominator,
                use_mov=False
            )
        
        # Calculate fitness
        if optimize_for == "accuracy":
            acc_all, acc_future, n_future = elo_accuracy(trial, cutoff)
            # Prefer future accuracy if available, otherwise use overall
            fitness = acc_future if acc_future is not None else acc_all
            if fitness is None:
                fitness = 0.0
        else:
            # For ROI optimization, we would need additional logic
            # For now, default to accuracy
            acc_all, acc_future, n_future = elo_accuracy(trial, cutoff)
            fitness = acc_future if acc_future is not None else acc_all
            if fitness is None:
                fitness = 0.0
        
        return fitness
    
    return fitness_function
