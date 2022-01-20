import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import linspace
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.model_selection import validation_curve
from sklearn.model_selection import learning_curve
import mlrose_hiive
import itertools
import dataframe_image as dfi

np.random.seed(51)


def file_to_path(fn, base_dir=None):
    """Return CSV file path for a given file name."""
    if base_dir is None:
        base_dir = os.environ.get("DATA_DIR", "./data/")
    return os.path.join(base_dir, "{}.csv".format(fn))


def get_data(filename, header=None):
    """Read csv data for a given csv file."""
    df = pd.read_csv(file_to_path(filename),
                     header=header,
                     na_values=["nan"],
                     )
    df = df.replace('?', np.nan)
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # adjust as binary classification
    if filename == 'winequality-white' or filename == 'winequality-red':
        y = y.astype(int)
        y[y < 6] = 0
        y[y >= 6] = 1

    if filename == 'stroke_data':
        x = df.iloc[1:, 1:-1]
        y = df.iloc[1:, -1]
        # gender
        x.iloc[:, 0][x.iloc[:, 0] == 'Male'] = 0
        x.iloc[:, 0][x.iloc[:, 0] == 'Female'] = 1
        x.iloc[:, 0][x.iloc[:, 0] == 'Other'] = 2
        # married
        x.iloc[:, 4][x.iloc[:, 4] == 'No'] = 0
        x.iloc[:, 4][x.iloc[:, 4] == 'Yes'] = 1
        # employment
        x.iloc[:, 5][x.iloc[:, 5] == 'children'] = 0
        x.iloc[:, 5][x.iloc[:, 5] == 'Govt_job'] = 1
        x.iloc[:, 5][x.iloc[:, 5] == 'Never_worked'] = 2
        x.iloc[:, 5][x.iloc[:, 5] == 'Private'] = 3
        x.iloc[:, 5][x.iloc[:, 5] == 'Self-employed'] = 4
        # residence type
        x.iloc[:, 6][x.iloc[:, 6] == 'Rural'] = 0
        x.iloc[:, 6][x.iloc[:, 6] == 'Urban'] = 1
        # bmi transform nan
        x.iloc[:, 8][x.iloc[:, 8] == True] = 0
        # smoking status
        x.iloc[:, 9][x.iloc[:, 9] == 'formerly smoked'] = 0
        x.iloc[:, 9][x.iloc[:, 9] == 'never smoked'] = 1
        x.iloc[:, 9][x.iloc[:, 9] == 'smokes'] = 2
        x.iloc[:, 9][x.iloc[:, 9] == 'Unknown'] = 3
        x = x.astype(float)
        y = y.astype(float)
    return x, y


def determine_train_and_query_times(learner, x_train, y_train, x_test):
    t0 = time.time()
    learner.clf.fit(x_train, y_train)
    t1 = time.time()
    t_time = t1 - t0
    print(f'Completed {learner.name} training in %f seconds' % t_time)
    t0 = time.time()
    _ = learner.clf.predict(x_test)
    t1 = time.time()
    q_time = t1 - t0
    print(f'Completed {learner.name} querying in %f seconds' % q_time)
    print('\n')
    return t_time, q_time


def gen_and_plot_validation_curve(learner, file, x_train, y_train, x_label, param_name, param_range):
    print(f"Generating validation curve for {learner.name}'s {param_name} param...")
    train_scores, test_scores = validation_curve(
        learner.base, x_train, y_train, param_name=param_name, param_range=param_range, verbose=10, cv=5)
    print(train_scores)
    print(test_scores)
    print(param_range)
    title = 'Validation Curve for ' + learner.name
    fig_name = learner.fig_prefix + "validation_curve_" + file + '_' \
               + str(datetime.datetime.now().isoformat()) + "_" + param_name + ".png"

    if param_name == 'hidden_layer_sizes':
        param_range = [1, 2, 3, 4, 5, 6]
    plt.figure()
    if param_name == 'C' or param_name == 'max_iter':
        plt.semilogx(param_range, np.mean(train_scores, axis=1), label='Training score')
        plt.semilogx(param_range, np.mean(test_scores, axis=1), label='Cross-validation score')
    else:
        plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Accuracy')
        plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Accuracy')
    plt.title(title)
    plt.xlabel(x_label)
    plt.legend(loc="best")
    plt.ylabel("Accuracy")
    plt.savefig(fig_name)
    plt.clf()


def gen_and_plot_learning_curve(learner, file, x_train, y_train):
    print(f"Generating learning curve for {learner.name}...")
    _, train_scores, test_scores = learning_curve(learner.clf, x_train, y_train, cv=5, n_jobs=-1)

    # plot the learning curve
    fig_name = learner.fig_prefix + "learning_curve_" + file + '_' + str(datetime.datetime.now().isoformat()) + ".png"
    title = learner.name + " Learner Accuracy vs. Training Set Size"
    plt.plot(linspace(0.1, 1.0, 5), np.mean(train_scores, axis=1), label='Training Accuracy')
    plt.plot(linspace(0.1, 1.0, 5), np.mean(test_scores, axis=1), label='Cross-validation Accuracy')
    plt.title(title)
    plt.xlabel('% of Training Set')
    plt.legend(loc="best")
    plt.ylabel("Accuracy")
    plt.savefig(fig_name)
    plt.clf()


def plot_training_times(times, y_pos, y_labels):
    plt.figure()
    plt.barh(y_pos, times)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(y_labels)
    plt.gca().invert_yaxis()
    plt.title('Model Training Time Comparisons')
    plt.xlabel('Training time (s)')
    fig_name = 'training_times_' + str(datetime.datetime.now().isoformat()) + ".png"
    plt.savefig(fig_name)
    plt.clf()


def plot_query_times(times, y_pos, y_labels):
    plt.figure()
    plt.barh(y_pos, times)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(y_labels)
    plt.gca().invert_yaxis()
    plt.title('Model Query Time Comparisons')
    plt.xlabel('Query time (s)')
    fig_name = 'query_times_' + str(datetime.datetime.now().isoformat()) + ".png"
    plt.savefig(fig_name)
    plt.clf()


def fit_base(learner, x_train, y_train, x_test, y_test):
    print(f"Fitting {learner.name} classifier...")
    learner.base.fit(x_train, y_train)
    y_pred = learner.base.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {learner.name} without hyperparameter tuning is %.2f%%' % (accuracy * 100))
    return learner.base


def tune_hyperparameters(learner, x_train, y_train, x_test, y_test):
    print(f"Tuning {learner.name} classifier with grid search...")
    learner.clf.fit(x_train, y_train)
    print(f"Best params for optimized {learner.name}: {learner.clf.best_params_}")
    print(f"Best score for optimized {learner.name}: {learner.clf.best_score_}")
    learner.base = learner.clf.best_estimator_
    learner.base.set_params(**learner.clf.best_params_)
    print("Setting base params to: ", learner.clf.best_params_)
    y_pred = learner.clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {learner.name} with hyperparameter tuning is %.2f%%' % (accuracy * 100))


def random_hill_climb_timed(problem_class, problem, max_attempts=100, max_iters=100, curve=True, random_state=51,
                            restarts=100, fevals=True):
    print(f"Running RHC for problem {problem_class.func_name}...")
    start_time = time.time()
    rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(
        problem=problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=curve,
        random_state=random_state,
        restarts=restarts,
    )
    end_time = time.time()
    rhc_time = end_time - start_time
    print("Time (s): {}".format(rhc_time) + '\n')
    return rhc_time, rhc_best_state, rhc_best_fitness, rhc_fitness_curve


def simulated_annealing_timed(problem_class, problem, max_attempts=100, max_iters=100, curve=True, random_state=51,
                              schedule=mlrose_hiive.GeomDecay(init_temp=1, decay=0.1, min_temp=1), fevals=True):
    print(f"Running SA for problem {problem_class.func_name}...")
    start_time = time.time()
    sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
        problem=problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=curve,
        random_state=random_state,
        schedule=schedule)
    end_time = time.time()
    sa_time = end_time - start_time
    print("Time (s): {}".format(sa_time) + '\n')
    return sa_time, sa_best_state, sa_best_fitness, sa_fitness_curve


def genetic_algorithm_timed(problem_class, problem, max_attempts=100, max_iters=100, curve=True, random_state=51,
                            pop_size=200, mutation_prob=0.2, fevals=True):
    print(f"Running GA for problem {problem_class.func_name}...")
    start_time = time.time()
    ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
        problem=problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=curve,
        random_state=random_state,
        pop_size=pop_size,
        mutation_prob=mutation_prob,
    )
    end_time = time.time()
    ga_time = end_time - start_time
    print("Time (s): {}".format(ga_time) + '\n')
    return ga_time, ga_best_state, ga_best_fitness, ga_fitness_curve


def mimic_timed(problem_class, problem, max_attempts=100, max_iters=100, curve=True, random_state=51, keep_pct=0.25):
    print(f"Running MIMIC for problem {problem_class.func_name}...")
    start_time = time.time()
    mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
        problem=problem,
        max_attempts=max_attempts,
        max_iters=max_iters,
        curve=curve,
        random_state=random_state,
        keep_pct=keep_pct
    )
    end_time = time.time()
    mimic_time = end_time - start_time
    print("Time (s): {}".format(mimic_time) + '\n')
    return mimic_time, mimic_best_state, mimic_best_fitness, mimic_fitness_curve


def assess_accuracy_of_hyperparams(clf, x_train, y_train, x_test, y_test, best_fitness_value, best_params, i, alg):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    base_final_score = accuracy_score(y_test, y_pred)
    if best_fitness_value == 0:
        best_params = i
        best_fitness_value = base_final_score
        print("Setting initial score of {} to {}, params: {}".format(alg, base_final_score, i))
    elif base_final_score > best_fitness_value:
        best_params = i
        best_fitness_value = base_final_score
        print("Accuracy of {} improved to {} with parameters {}".format(alg, base_final_score, best_params))
    return best_params, best_fitness_value


def find_opt_hyperparams(problem_class, run_nn=False, algorithm='gradient_descent', x_train=[], y_train=[], x_test=[],
                         y_test=[]):
    results = {}
    best_params_rhc = None
    best_fitness_value_rhc = 0
    best_params_sa = None
    best_fitness_value_sa = 0
    best_params_ga = None
    best_fitness_value_ga = 0
    best_params_mimic = None
    best_fitness_value_mimic = 0
    alg_settings = {
        'RHC': {
            'algorithm': mlrose_hiive.random_hill_climb,
            'hyperparameters': [
                [1, 20, 50, 70, 100],  # length
                [0, 25, 75, 100],  # restarts
                [100, 200, 300, 400, 500, 600]  # max attempts/iters
            ],
        },
        'SA': {
            'algorithm': mlrose_hiive.simulated_annealing,
            'hyperparameters': [
                [1, 2, 4, 8, 16, 32, 64],  # initial temp
                [0.1, 0.2, 0.4, 0.8],  # decay
                [0.001, 0.01, 0.1, 1],  # min temp
                [100, 200, 300, 400, 500, 600],  # max attempts/iters
                [1, 20, 50, 70, 100],  # length
            ],
        },
        'GA': {
            'algorithm': mlrose_hiive.genetic_alg,
            'hyperparameters': [
                [100, 200, 400],  # pop size
                [0.2, 0.4, 0.8],  # mutation prob
                [100, 200, 300, 400, 500, 600],  # max attempts/iters
                [1, 20, 50, 70, 100],  # length
            ],
        },
        'MIMIC': {
            'algorithm': mlrose_hiive.mimic,
            'hyperparameters': [
                [1, 20, 50, 70, 100],  # length
                [0, 25, 75, 100],  # max attempts/iters
            ]
        },
        'NN': {
            'algorithm': mlrose_hiive.NeuralNetwork,
            'hyperparameters': [
                [0, 50],  # RHC restarts
                [1, 2, 4, 8, 16, 32, 64],  # SA initial temp
                [0.1, 0.2, 0.4, 0.8],  # SA decay
                [0.001, 0.01, 0.1, 1],  # SA min temp
                [100, 200, 400],  # GA pop size
                [0.2, 0.4, 0.8],  # GA mutation prob
                [100, 300, 500, 600],  # max attempts/iters
                [1, 50, 100],  # length
                [0.0001, 0.001, 0.01],  # learning rate
                ['tanh', 'relu', 'sigmoid']  # activation
            ],
        }
    }
    for alg, conf in alg_settings.items():
        if not run_nn:
            print(f'Determining optimal hyperparams for problem {problem_class.func_name} using {alg}...')
        if not run_nn and alg == "NN":
            continue
        elif not run_nn and alg == 'RHC':
            best_restart_param = None
            best_restart_fitness_value = None
            for i in itertools.product(*conf['hyperparameters']):
                problem = mlrose_hiive.DiscreteOpt(length=i[0], fitness_fn=problem_class.fitness_func, maximize=True)
                rhc_best_state, rhc_best_fitness, rhc_fitness_curve = mlrose_hiive.random_hill_climb(problem,
                                                                                                     max_attempts=i[2],
                                                                                                     max_iters=i[2],
                                                                                                     curve=True,
                                                                                                     random_state=51,
                                                                                                     restarts=i[1])
                if not best_restart_fitness_value:
                    best_restart_param = i
                    best_restart_fitness_value = rhc_best_fitness
                elif rhc_best_fitness > best_restart_fitness_value:
                    best_restart_param = i
                    best_restart_fitness_value = rhc_best_fitness
            print("Best RHC parameters for {} = {}".format(alg, str(best_restart_param)))
            results[alg] = {
                'best_params': best_restart_param,
                'best_fitness': best_restart_fitness_value
            }
        elif not run_nn and alg == 'SA':
            best_params = None
            best_fitness_value = None
            for i in itertools.product(*conf['hyperparameters']):
                problem = mlrose_hiive.DiscreteOpt(length=i[4], fitness_fn=problem_class.fitness_func, maximize=True)
                decay = mlrose_hiive.GeomDecay(init_temp=i[0], decay=i[1], min_temp=i[2])
                sa_best_state, sa_best_fitness, sa_fitness_curve = mlrose_hiive.simulated_annealing(
                    problem,
                    max_attempts=i[3],
                    max_iters=i[3],
                    curve=True,
                    random_state=51,
                    schedule=decay)
                if not best_fitness_value:
                    best_params = i
                    best_fitness_value = sa_best_fitness
                elif sa_best_fitness > best_fitness_value:
                    best_params = i
                    best_fitness_value = sa_best_fitness
            print("Best SA parameters for {} = {}".format(alg, str(best_params)))
            results[alg] = {
                'best_params': best_params,
                'best_fitness': best_fitness_value
            }
        elif not run_nn and alg == 'GA':
            best_params = None
            best_fitness_value = None
            for i in itertools.product(*conf['hyperparameters']):
                problem = mlrose_hiive.DiscreteOpt(length=i[3], fitness_fn=problem_class.fitness_func, maximize=True)
                ga_best_state, ga_best_fitness, ga_fitness_curve = mlrose_hiive.genetic_alg(
                    problem,
                    max_attempts=i[2],
                    max_iters=i[2],
                    curve=True,
                    random_state=51,
                    pop_size=i[0],
                    mutation_prob=i[1])
                if not best_fitness_value:
                    best_params = i
                    best_fitness_value = ga_best_fitness
                elif ga_best_fitness > best_fitness_value:
                    best_params = i
                    best_fitness_value = ga_best_fitness
            print("Best parameters for {} = {}".format(alg, str(best_params)))
            results[alg] = {
                'best_params': best_params,
                'best_fitness': best_fitness_value
            }
        elif alg == "NN":
            print(f'Determining optimal hyperparams for NN using {algorithm}...')
            # evaluate all parameters
            for i in itertools.product(*conf['hyperparameters']):
                # for each algorithm
                for alg, conf in alg_settings.items():
                    if alg == "NN":
                        continue
                    elif alg == "RHC":
                        clf = mlrose_hiive.NeuralNetwork(
                            hidden_nodes=[10, 2],
                            algorithm=algorithm,
                            bias=True,
                            max_attempts=i[6],
                            max_iters=i[6],
                            curve=True,
                            random_state=51,
                            restarts=i[0],
                            learning_rate=i[8],
                            activation=i[9],
                            early_stopping=True,
                        )
                        best_params_rhc, best_fitness_value_rhc = assess_accuracy_of_hyperparams(clf, x_train, y_train,
                                                                                                 x_test, y_test,
                                                                                                 best_fitness_value_rhc,
                                                                                                 best_params_rhc, i,
                                                                                                 alg)
                    elif alg == "SA":
                        decay = mlrose_hiive.GeomDecay(init_temp=i[1], decay=i[2], min_temp=i[3])
                        clf = mlrose_hiive.NeuralNetwork(
                            hidden_nodes=[10, 10],
                            algorithm=algorithm,
                            bias=True,
                            max_attempts=i[6],
                            max_iters=i[6],
                            curve=True,
                            random_state=51,
                            schedule=decay,
                            learning_rate=i[8],
                            activation=i[9],
                            early_stopping=True,
                            restarts=i[0],
                        )
                        best_params_sa, best_fitness_value_sa = assess_accuracy_of_hyperparams(clf, x_train, y_train,
                                                                                               x_test, y_test,
                                                                                               best_fitness_value_sa,
                                                                                               best_params_sa, i, alg)
                    elif alg == "GA":
                        clf = mlrose_hiive.NeuralNetwork(
                            hidden_nodes=[10, 10],
                            algorithm=algorithm,
                            bias=True,
                            max_attempts=i[6],
                            max_iters=i[6],
                            curve=True,
                            random_state=51,
                            pop_size=i[4],
                            learning_rate=i[8],
                            activation=i[9],
                            mutation_prob=i[5],
                            early_stopping=True,
                            restarts=i[0],
                        )
                        best_params_ga, best_fitness_value_ga = assess_accuracy_of_hyperparams(clf, x_train, y_train,
                                                                                               x_test, y_test,
                                                                                               best_fitness_value_ga,
                                                                                               best_params_ga, i, alg)
                    elif alg == "MIMIC":
                        clf = mlrose_hiive.NeuralNetwork(
                            hidden_nodes=[10, 10],
                            algorithm=algorithm,
                            bias=True,
                            max_attempts=
                            i[6],
                            max_iters=
                            i[6],
                            curve=True,
                            learning_rate=i[8],
                            activation=i[9],
                            random_state=51,
                            early_stopping=True,
                            restarts=i[0],
                        )
                        best_params_mimic, best_fitness_value_mimic = assess_accuracy_of_hyperparams(clf, x_train,
                                                                                                     y_train, x_test,
                                                                                                     y_test,
                                                                                                     best_fitness_value_mimic,
                                                                                                     best_params_mimic,
                                                                                                     i, alg)
            results[algorithm] = {
                'best_params_rhc': best_params_rhc,
                'best_fitness_rhc': best_fitness_value_rhc,
                'best_params_sa': best_params_sa,
                'best_fitness_sa': best_fitness_value_sa,
                'best_params_ga': best_params_ga,
                'best_fitness_ga': best_fitness_value_ga,
                'best_params_mimic': best_params_mimic,
                'best_fitness_mimic': best_fitness_value_mimic,
            }
            return results
        elif not run_nn and alg == "MIMIC":
            best_params = None
            best_fitness_value = None
            for i in itertools.product(*conf['hyperparameters']):
                problem = mlrose_hiive.DiscreteOpt(length=i[0], fitness_fn=problem_class.fitness_func, maximize=True)
                mimic_best_state, mimic_best_fitness, mimic_fitness_curve = mlrose_hiive.mimic(
                    problem,
                    max_attempts=i[1],
                    max_iters=i[1],
                    curve=True,
                    random_state=51,
                )
                if not best_fitness_value:
                    best_params = i
                    best_fitness_value = mimic_best_fitness
                elif mimic_best_fitness > best_fitness_value:
                    best_params = i
                    best_fitness_value = mimic_best_fitness
            print("Best parameters for {} = {}".format(alg, str(best_params)))
            results[alg] = {
                'best_params': best_params,
                'best_fitness': best_fitness_value
            }
    return results


def run_neural_net(alg, x_train, y_train, x_test, y_test, algorithm='gradient_descent', alpha=1.5,
                   hidden_nodes=[10, 10], activation='relu',
                   max_attempts=1000, max_iters=5000):
    print(f'Training and testing optimized NN for alg {algorithm}...')
    clf = mlrose_hiive.NeuralNetwork(
        algorithm=algorithm,
        bias=True,
        hidden_nodes=hidden_nodes,
        activation=activation,
        early_stopping=True,
        max_attempts=max_attempts,
        max_iters=max_iters,
        learning_rate=alpha,
        restarts=0,
        curve=True,
        random_state=51,
    )

    if alg == "SA":
        clf.schedule = mlrose_hiive.GeomDecay(init_temp=1, decay=0.1, min_temp=0.001)

    if alg == "GA":
        clf.pop_size = 100
        clf.mutation_prob = 0.2

    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()
    diff = end_time - start_time
    y_pred = clf.predict(x_test)
    base_final_score = accuracy_score(y_test, y_pred)
    return diff, base_final_score, clf.fitness_curve


def plot_fitness_graph(problem_class, rhc_fc, sa_fc, ga_fc, m_fc):
    iterations = range(1, problem_class.max_attempts + 1)
    plt.clf()
    plt.figure()
    plt.plot(iterations, rhc_fc[:, 0], label='RHC')
    plt.plot(iterations, sa_fc[:, 0], label='SA')
    plt.plot(iterations, ga_fc[:, 0], label='GA')
    plt.plot(iterations, m_fc[:, 0], label='MIMIC')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title("Fitness vs Iterations for %s" % problem_class.name)
    plt.savefig(f"{problem_class.func_name}_fitness_{str(datetime.datetime.now().isoformat())}.png")
    plt.clf()


def plot_iter_length(problem_class, length_set, iter_rhc, iter_sa, iter_ga, iter_m):
    plt.clf()
    plt.figure()
    plt.plot(length_set, iter_rhc, label='RHC')
    plt.plot(length_set, iter_sa, label='SA')
    plt.plot(length_set, iter_ga, label='GA')
    plt.plot(length_set, iter_m, label='MIMIC')
    plt.legend(loc="best")
    plt.xlabel("Bitstring Length")
    plt.ylabel("Evaluations Per Iteration")
    plt.title("Evaluations Per Iteration vs Bitstring Length for %s" % problem_class.name)
    plt.savefig(f"{problem_class.func_name}_fevals_{str(datetime.datetime.now().isoformat())}.png")
    plt.clf()


def plot_run_times(problem_class, times, y_pos, y_labels):
    plt.clf()
    plt.figure()
    plt.barh(y_pos, times)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(y_labels)
    plt.gca().invert_yaxis()
    plt.title('Algorithm')
    plt.xlabel('Search time (s)')
    fig_name = problem_class.func_name + '_run_times_' + str(datetime.datetime.now().isoformat()) + ".png"
    plt.savefig(fig_name)
    plt.clf()


def plot_nn_run_times(times, y_pos, y_labels):
    plt.clf()
    plt.figure()
    plt.barh(y_pos, times)
    plt.gca().set_yticks(y_pos)
    plt.gca().set_yticklabels(y_labels)
    plt.gca().invert_yaxis()
    plt.title('Algorithm with Neural Net')
    plt.xlabel('Run time (s)')
    fig_name = 'neural_net_run_times_' + str(datetime.datetime.now().isoformat()) + ".png"
    plt.savefig(fig_name)
    plt.clf()


def plot_loss(alg, fc):
    plt.plot(fc)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"nn_{alg}_loss_{str(datetime.datetime.now().isoformat())}.png")
    plt.clf()


def plot_alg_runtimes(alg, rhc, sa, ga, mimic):
    data = [('RHC', round(rhc, 5)),
            ('SA', round(sa, 5)),
            ('GA', round(ga, 5)),
            ('MIMIC', round(mimic, 5))]

    df = pd.DataFrame(data, columns=['Algorithm', 'Time (s)'])
    dfi.export(df, "%s_times.png" % alg)
