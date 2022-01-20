import mlrose_hiive
from utils import random_hill_climb_timed, simulated_annealing_timed \
    , genetic_algorithm_timed, mimic_timed


# problem best solved with MIMIC:
# {'RHC': {'best_params': (100, 0, 500), 'best_fitness': 100.0},
# 'SA': {'best_params': (1, 0.1, 0.001, 500, 100), 'best_fitness': 100.0},
# 'GA': {'best_params': (100, 0.2, 100, 100), 'best_fitness': 100.0},
# 'MIMIC': {'best_params': (100, 25), 'best_fitness': 100.0}}
class TravellingSales:
    def __init__(self, max_attempts=100, max_iterations=100):
        self.name = "Travelling Sales"
        self.func_name = 'travelling_sales'
        self.coords_list = [(1, 1), (4, 2), (5, 2), (6, 4), (4, 4), (3, 6), (1, 5), (2, 3)]
        self.dist_list = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426),
                          (0, 5, 5.3852), (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000),
                          (1, 3, 2.8284), (1, 4, 2.0000), (1, 5, 4.1231), (1, 6, 4.2426),
                          (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), (2, 5, 4.4721),
                          (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056),
                          (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623),
                          (4, 7, 2.2361), (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]
        self.fitness_func = mlrose_hiive.TravellingSales(coords=self.coords_list, distances=self.dist_list)
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
        return mimic_timed(problem_class=self, problem=problem, max_attempts=self.max_attempts,
                           max_iters=self.max_iterations)
