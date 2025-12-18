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
import pandas as pd
from typing import List, Dict, Callable, Tuple, Optional
import copy
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        mutation_std_ratio: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        selection_method: str = "tournament",
        crossover_method: str = "uniform",
        random_seed: Optional[int] = None,
        verbose: bool = True,
        optimization_mode: Optional[str] = None
    ):
        """
        Initialize the genetic algorithm.
        
        Args:
            param_bounds: Dict mapping parameter names to (min, max) tuples
            fitness_fn: Function that takes a parameter dict and returns fitness score
            population_size: Number of individuals in the population
            elite_size: Number of best individuals to preserve each generation
            mutation_rate: Probability of mutation for each gene
            mutation_std_ratio: Ratio of parameter range to use as mutation std dev (default 0.1)
            crossover_rate: Probability of crossover between parents
            tournament_size: Size of tournament for tournament selection
            selection_method: "tournament" or "roulette"
            crossover_method: "single", "two_point", or "uniform"
            random_seed: Random seed for reproducibility
            verbose: Print progress messages
            optimization_mode: Optional optimization mode ("log_loss", etc.) for display formatting
        """
        self.param_bounds = param_bounds
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.mutation_std_ratio = mutation_std_ratio
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.verbose = verbose
        self.optimization_mode = optimization_mode
        
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
                # Gaussian mutation with configurable standard deviation
                std_dev = (max_val - min_val) * self.mutation_std_ratio
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
    
    def _format_metric_for_display(self, fitness: float) -> Tuple[str, float]:
        """
        Format fitness value for display based on optimization mode.
        
        Args:
            fitness: The fitness value to format
            
        Returns:
            Tuple of (metric_name, display_value) where:
            - For log_loss mode: converts fitness back to log_loss (lower is better)
            - For other modes: returns fitness as-is (higher is better)
        """
        if self.optimization_mode == "log_loss":
            # Convert fitness back to log_loss: log_loss = -ln(fitness)
            # Clamp fitness to prevent log(0) which would be undefined.
            # Using 1e-15 as minimum allows displaying log_loss up to ~34.5,
            # which is well beyond any practical prediction scenario
            # (perfect predictions have log_loss=0, random guessing≈0.693)
            fitness_clamped = max(fitness, 1e-15)
            log_loss_value = -np.log(fitness_clamped)
            return ("Log Loss", log_loss_value)
        else:
            return ("Fitness", fitness)
    
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
                # Format display values based on optimization mode
                metric_name, best_display = self._format_metric_for_display(best_fitness)
                _, avg_display = self._format_metric_for_display(avg_fitness)
                _, worst_display = self._format_metric_for_display(worst_fitness)
                
                print(f"Generation {gen + 1}/{generations}: "
                      f"Best={best_display:.6f}, Avg={avg_display:.6f}, "
                      f"Worst={worst_display:.6f}")
            
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
            
            # Format final metric display
            metric_name, final_display = self._format_metric_for_display(self.best_individual.fitness)
            print(f"Best {metric_name.lower()}: {final_display:.6f}")
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
        return pd.DataFrame(self.history)


def create_elo_fitness_function(
    df,
    test_df=None,
    base_elo: float = 1500,
    use_validation_split: bool = True,
    validation_percentile: float = 0.8,
    optimize_for: str = "accuracy",
    fitness_weights: Optional[Dict[str, float]] = None
) -> Callable[[Dict[str, float]], float]:
    """
    Create a fitness function for Elo parameter optimization.
    
    This function returns a fitness function that can be used with the genetic
    algorithm to optimize Elo parameters using multiple metrics.
    
    Args:
        df: Training DataFrame with fight data
        test_df: Optional test DataFrame for out-of-sample evaluation
        base_elo: Base Elo rating for new fighters
        use_validation_split: Whether to use time-based validation split
        validation_percentile: Percentile for validation split
        optimize_for: "accuracy" (default), "roi", "log_loss", or "composite"
            - "accuracy": Simple prediction accuracy only (higher is better, range: 0-1)
            - "roi": Return on investment only (higher is better, range: -1 to 1)
            - "log_loss": Logarithmic loss only (lower is better, converted to higher-is-better for GA)
            - "composite": Weighted combination of all metrics
        fitness_weights: Optional dict for composite fitness. Keys: 
            'accuracy', 'log_loss', 'brier_score', 'roi'
            Default: {'accuracy': 0.3, 'log_loss': 0.2, 'brier_score': 0.2, 'roi': 0.3}
        
    Returns:
        Fitness function that takes parameter dict and returns fitness score
    """
    from optimization.optimal_k_with_mov import run_basic_elo, elo_accuracy, compute_fight_predictions
    import numpy as np
    
    # Default fitness weights for composite mode
    if fitness_weights is None:
        fitness_weights = {
            'accuracy': 0.3,
            'log_loss': 0.2, 
            'brier_score': 0.2,
            'roi': 0.3
        }
    
    if use_validation_split:
        cutoff = df["DATE"].quantile(validation_percentile)
    else:
        cutoff = None
    
    def fitness_function(params: Dict[str, float]) -> float:
        """Evaluate fitness for a parameter set using multiple metrics."""
        # Extract parameters with defaults
        k = params.get("k", 32)
        denominator = params.get("denominator", 400)
        confidence_threshold = params.get("confidence_threshold", 50)
        
        # MOV weights (use defaults if not in params)
        use_mov = any(key.startswith("w_") for key in params.keys())
        
        # Run Elo with these parameters
        try:
            trial = run_basic_elo(
                df.copy(),
                k=k,
                base_elo=base_elo,
                denominator=denominator,
                use_mov=use_mov
            )
        except Exception:
            return 0.0
        
        # Get predictions on validation set
        if cutoff is not None:
            val_df = trial[trial["DATE"] > cutoff].copy()
        else:
            val_df = trial.copy()
        
        # Extract predictions and actuals with filtering
        predictions = []
        actuals = []
        for _, row in val_df.iterrows():
            if row.get("result") not in (0, 1):
                continue
            if pd.isna(row.get("DATE")):
                continue
            if row.get("precomp_elo") == row.get("opp_precomp_elo"):
                continue
            
            # Check bout counts
            bout1 = row.get("precomp_boutcount", 0)
            bout2 = row.get("opp_precomp_boutcount", 0)
            if pd.isna(bout1) or pd.isna(bout2) or bout1 < 1 or bout2 < 1:
                continue
            
            # Calculate prediction probability
            elo_diff = row["precomp_elo"] - row["opp_precomp_elo"]
            pred_prob = 1.0 / (1.0 + 10.0 ** (-elo_diff / denominator))
            
            predictions.append(pred_prob)
            actuals.append(int(row["result"]))
        
        if len(predictions) == 0:
            # No valid predictions means the parameters are invalid
            # Return worst possible fitness
            if optimize_for == "roi":
                return -1.0  # Worst possible ROI
            elif optimize_for == "log_loss":
                return 1e-15  # Near-zero fitness (very high log loss)
            else:  # accuracy or composite
                return 0.0  # Worst accuracy
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate different metrics
        if optimize_for == "accuracy":
            # Simple accuracy
            pred_labels = (predictions > 0.5).astype(int)
            fitness = np.mean(pred_labels == actuals)
        
        elif optimize_for == "roi":
            # Realistic ROI calculation with implied odds based on prediction probabilities
            total_profit = 0.0
            total_bets = 0
            for pred, actual in zip(predictions, actuals):
                # Use prediction confidence (distance from 0.5) to determine if we should bet
                confidence = abs(pred - 0.5) * 2  # Scale to 0-1 range
                # Scale by 1000 to match typical Elo difference magnitude for threshold comparison
                # e.g., confidence_threshold=50 means bet when confidence > 0.05 (5%)
                if confidence * 1000 >= confidence_threshold:
                    # Determine predicted winner and their win probability
                    pred_winner = 1 if pred > 0.5 else 0
                    pred_prob = pred if pred > 0.5 else (1 - pred)
                    
                    # Calculate realistic payout based on implied odds
                    # If pred_prob = 0.7 (70% favorite): payout = 1/0.7 - 1 ≈ 0.43 (bet $1 to win $0.43)
                    # If pred_prob = 0.3 (30% underdog): payout = 1/0.3 - 1 ≈ 2.33 (bet $1 to win $2.33)
                    # Clamp probability to avoid division by very small numbers
                    pred_prob_clamped = max(pred_prob, 0.01)  # Minimum 1% probability (max 99:1 odds)
                    payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
                    
                    # We always bet 1 unit
                    if pred_winner == actual:
                        # Win: get back bet + payout
                        total_profit += payout_multiplier
                    else:
                        # Lose: lose the bet
                        total_profit -= 1.0
                    total_bets += 1
            
            # If no bets were made, return worst possible ROI (-1.0)
            # This ensures models that make no predictions are penalized
            # rather than appearing as "break-even" (0.0)
            roi = (total_profit / total_bets) if total_bets > 0 else -1.0
            fitness = roi
        
        elif optimize_for == "log_loss":
            # Log Loss (lower is better, so we invert it for GA maximization)
            eps = 1e-15
            predictions_clipped = np.clip(predictions, eps, 1 - eps)
            log_loss = np.mean(
                -(actuals * np.log(predictions_clipped) +
                  (1 - actuals) * np.log(1 - predictions_clipped))
            )
            # Convert to fitness (higher is better)
            # Use exponential decay to map log_loss to fitness in range (0, 1]
            # Perfect: log_loss=0 -> fitness=1.0
            # Random: log_loss=ln(2)≈0.693 -> fitness≈0.5 (theoretical binary classifier random guess)
            # Poor: log_loss>0.693 -> fitness<0.5
            # This allows GA to distinguish between different "bad" models
            fitness = np.exp(-log_loss)
        
        else:  # composite
            # Accuracy component
            pred_labels = (predictions > 0.5).astype(int)
            accuracy = np.mean(pred_labels == actuals)
            
            # Log Loss component (lower is better, so invert and scale)
            eps = 1e-15
            predictions_clipped = np.clip(predictions, eps, 1 - eps)
            log_loss = np.mean(
                -(actuals * np.log(predictions_clipped) +
                  (1 - actuals) * np.log(1 - predictions_clipped))
            )
            # Use exponential mapping: log_loss=0 -> 1.0, log_loss=0.693 -> ~0.5
            log_loss_score = np.exp(-log_loss)
            
            # Brier Score component (lower is better, so invert)
            brier_score = np.mean((predictions - actuals) ** 2)
            # Scale: perfect = 0, random = 0.25, invert to make higher better
            brier_score_score = max(0, 1.0 - brier_score / 0.25)
            
            # ROI component
            total_profit = 0
            total_bets = 0
            # Calculate Elo differences for confidence filtering
            elo_diffs = []
            for i in range(len(predictions)):
                # We need to recalculate elo_diff from the row data
                # For now, use prediction confidence as proxy
                elo_diffs.append(abs(predictions[i] - 0.5) * 1000)  # Scale to Elo points
            
            for i, (pred, actual, elo_diff_val) in enumerate(zip(predictions, actuals, elo_diffs)):
                if elo_diff_val >= confidence_threshold:
                    # Determine predicted winner and their win probability
                    pred_winner = 1 if pred > 0.5 else 0
                    pred_prob = pred if pred > 0.5 else (1 - pred)
                    
                    # Calculate realistic payout based on implied odds
                    # Clamp probability to avoid division by very small numbers
                    pred_prob_clamped = max(pred_prob, 0.01)
                    payout_multiplier = (1.0 / pred_prob_clamped) - 1.0
                    
                    if pred_winner == actual:
                        total_profit += payout_multiplier
                    else:
                        total_profit -= 1.0
                    total_bets += 1
            
            # If no bets were made, return worst possible ROI (-1.0)
            # This ensures models that make no predictions are penalized
            roi = (total_profit / total_bets) if total_bets > 0 else -1.0
            # Scale ROI: -1.0 to 1.0 -> 0.0 to 1.0
            roi_score = (roi + 1.0) / 2.0
            
            # Combine with weights
            fitness = (
                fitness_weights.get('accuracy', 0.3) * accuracy +
                fitness_weights.get('log_loss', 0.2) * log_loss_score +
                fitness_weights.get('brier_score', 0.2) * brier_score_score +
                fitness_weights.get('roi', 0.3) * roi_score
            )
        
        return fitness
    
    return fitness_function
