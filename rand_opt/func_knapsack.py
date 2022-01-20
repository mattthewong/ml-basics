import mlrose_hiive
from utils import random_hill_climb_timed, simulated_annealing_timed \
    , genetic_algorithm_timed, mimic_timed


class KnapSack:
    def __init__(self, max_attempts=100, max_iterations=100,
                 max_weight_pct=0.6):
        self.name = "Knapsack"
        self.func_name = 'knapsack'
        self.fitness_func = mlrose_hiive.Knapsack(
            weights=[10, 5, 2, 8, 15, 11, 4, 7, 1, 20],
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            max_weight_pct=max_weight_pct)
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def set_configs(self, max_attempts, max_iterations):
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def run_random_hill_climb(self):
        problem = mlrose_hiive.DiscreteOpt(length=10, fitness_fn=self.fitness_func, maximize=True, max_val=2)
        return random_hill_climb_timed(problem_class=self, problem=problem, restarts=0, max_iters=self.max_iterations,
                                       max_attempts=self.max_attempts,
                                       )

    def run_simulated_annealing(self):
        problem = mlrose_hiive.DiscreteOpt(length=10, fitness_fn=self.fitness_func, maximize=True, max_val=2)
        return simulated_annealing_timed(problem_class=self, problem=problem,
                                         schedule=mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.1, min_temp=0.001)
                                         , max_iters=self.max_iterations,
                                         max_attempts=self.max_attempts,
                                         )

    def run_genetic_algorithm(self):
        problem = mlrose_hiive.DiscreteOpt(length=10, fitness_fn=self.fitness_func, maximize=True, max_val=2)
        return genetic_algorithm_timed(problem_class=self, problem=problem, pop_size=100, mutation_prob=0.2,
                                       max_attempts=self.max_attempts, max_iters=self.max_iterations)

    def run_mimic(self):
        problem = mlrose_hiive.DiscreteOpt(length=10, fitness_fn=self.fitness_func, maximize=True, max_val=2)
        return mimic_timed(problem_class=self, problem=problem, max_attempts=self.max_attempts,
                           max_iters=self.max_iterations)
