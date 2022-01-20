import mlrose_hiive
from utils import random_hill_climb_timed, simulated_annealing_timed \
    , genetic_algorithm_timed, mimic_timed


# problem best solved with Genetic Algorithm:
# {'RHC': {'best_params': (70, 100, 600), 'best_fitness': 104.0},
# 'SA': {'best_params': (1, 0.1, 0.001, 500, 50), 'best_fitness': 68.0},
# 'GA': {'best_params': (100, 0.2, 400, 100), 'best_fitness': 189.0},
# 'MIMIC': {'best_params': (100, 25), 'best_fitness': 122.0}}
class SixPeaks:
    def __init__(self, max_attempts=100, max_iterations=100):
        self.name = "Six Peaks"
        self.func_name = 'six_peaks'
        self.fitness_func = mlrose_hiive.SixPeaks()
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def set_configs(self, max_attempts, max_iterations):
        self.max_attempts = max_attempts
        self.max_iterations = max_iterations

    def run_random_hill_climb(self):
        problem = mlrose_hiive.DiscreteOpt(length=70, fitness_fn=self.fitness_func, maximize=True)
        return random_hill_climb_timed(problem_class=self, problem=problem, restarts=100, max_iters=self.max_iterations,
                                       max_attempts=self.max_attempts
                                       )

    def run_simulated_annealing(self):
        problem = mlrose_hiive.DiscreteOpt(length=50, fitness_fn=self.fitness_func, maximize=True)
        return simulated_annealing_timed(problem_class=self, problem=problem,
                                         schedule=mlrose_hiive.GeomDecay(init_temp=1.0, decay=0.1, min_temp=0.001),
                                         max_iters=self.max_iterations,
                                         max_attempts=self.max_attempts
                                         )

    def run_genetic_algorithm(self):
        problem = mlrose_hiive.DiscreteOpt(length=100, fitness_fn=self.fitness_func, maximize=True)
        return genetic_algorithm_timed(problem_class=self, problem=problem, pop_size=100, mutation_prob=0.2,
                                       max_attempts=self.max_attempts, max_iters=self.max_iterations)

    def run_mimic(self):
        problem = mlrose_hiive.DiscreteOpt(length=25, fitness_fn=self.fitness_func, maximize=True)
        return mimic_timed(problem_class=self, problem=problem, max_attempts=self.max_attempts, max_iters=self.max_iterations)
