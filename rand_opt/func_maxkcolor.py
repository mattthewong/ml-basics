import mlrose_hiive
from utils import random_hill_climb_timed, simulated_annealing_timed \
    , genetic_algorithm_timed, mimic_timed


# problem best solved with MIMIC:
# {'RHC': {'best_params': (100, 0, 500), 'best_fitness': 100.0},
# 'SA': {'best_params': (1, 0.1, 0.001, 500, 100), 'best_fitness': 100.0},
# 'GA': {'best_params': (100, 0.2, 100, 100), 'best_fitness': 100.0},
# 'MIMIC': {'best_params': (100, 25), 'best_fitness': 100.0}}
class MaxKColor:
    def __init__(self, max_attempts=100, max_iterations=100):
        self.name = "MaxKColor"
        self.func_name = 'max_k_color'
        self.edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        self.fitness_func = mlrose_hiive.MaxKColor(edges=self.edges)
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def set_configs(self, max_attempts, max_iterations):
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def run_random_hill_climb(self):
        problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=self.fitness_func, maximize=True)
        return random_hill_climb_timed(problem_class=self, problem=problem, restarts=0, max_iters=self.max_iterations,
                                       max_attempts=self.max_attempts
                                       )

    def run_simulated_annealing(self):
        problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=self.fitness_func, maximize=True)
        return simulated_annealing_timed(problem_class=self, problem=problem,
                                         schedule=mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.1, min_temp=0.001),
                                         max_attempts=self.max_attempts, max_iters=self.max_iterations
                                         )

    def run_genetic_algorithm(self):
        problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=self.fitness_func, maximize=True)
        return genetic_algorithm_timed(problem_class=self, problem=problem, pop_size=100, mutation_prob=0.2,
                                       max_attempts=self.max_attempts, max_iters=self.max_iterations)

    def run_mimic(self):
        problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=self.fitness_func, maximize=True)
        return mimic_timed(problem_class=self, problem=problem, max_attempts=self.max_attempts, max_iters=self.max_iterations)
