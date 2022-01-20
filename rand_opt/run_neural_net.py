import sys
from numpy import nan, zeros, arange
from utils import get_data, plot_loss, run_neural_net, plot_nn_run_times, find_opt_hyperparams
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


def main():
    if len(sys.argv) != 1:
        print("Usage: python run_neural_net.py")
        sys.exit(1)

    # get data
    x, y = get_data("stroke_data")
    imp_mean = SimpleImputer(missing_values=nan, strategy='mean')
    imp_mean = imp_mean.fit(x)
    x = imp_mean.transform(x)

    x = preprocessing.scale(x)

    # seed
    rs = 51

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=rs)

    print("Features: ", x_train.shape[1])
    print("Samples for training: ", x_train.shape[0])
    print("Samples for testing: ", x_test.shape[0])

    # Control setting boolean flags
    run_gradient_descent = True  # run gradient descent
    run_random_hill_climb = False  # run random hill climb
    run_simulated_annealing = False  # run simulated annealing
    run_genetic_algorithm = False  # run genetic algorithm
    tune_neural_nets = False  # tune neural nets
    plot_loss_curve = True  # plot fitness curve
    plot_times = False  # plot train and query times

    times = zeros(4)
    algs = ('GD', 'RHC', 'SA', 'GA')

    # Gradient Descent
    if run_gradient_descent:
        if tune_neural_nets:
            results = find_opt_hyperparams(problem_class=None, run_nn=True, algorithm='gradient_descent',
                                           x_train=x_train,
                                           y_train=y_train, x_test=x_test, y_test=y_test)
            print(results)
        time, score, fitness_curve = run_neural_net("GD", x_train, y_train, x_test, y_test, 'gradient_descent')
        times[0] = time
        if plot_loss_curve:
            print("GD: ", fitness_curve.shape)
            fitness_curve = fitness_curve*-1
            plot_loss('gradient_descent', fitness_curve)

    # Random Hill Climb
    if run_random_hill_climb:
        if tune_neural_nets:
            results = find_opt_hyperparams(problem_class=None, run_nn=True, algorithm='random_hill_climb',
                                           x_train=x_train,
                                           y_train=y_train, x_test=x_test, y_test=y_test)
            print(results)
        time, score, fitness_curve = run_neural_net("RHC", x_train, y_train, x_test, y_test, 'random_hill_climb')
        times[1] = time
        if plot_loss_curve:
            print("RHC: ", fitness_curve.shape)
            plot_loss('random_hill_climb', fitness_curve[:, 0])

    # Simulated Annealing
    if run_simulated_annealing:
        if tune_neural_nets:
            results = find_opt_hyperparams(problem_class=None, run_nn=True, algorithm='simulated_annealing',
                                           x_train=x_train,
                                           y_train=y_train, x_test=x_test, y_test=y_test)
            print(results)
        time, score, fitness_curve = run_neural_net("SA", x_train, y_train, x_test, y_test, 'simulated_annealing')
        times[2] = time
        if plot_loss_curve:
            print("SA: ", fitness_curve.shape)
            plot_loss('simulated_annealing', fitness_curve[:, 0])

    # Genetic Algorithm
    if run_genetic_algorithm:
        if tune_neural_nets:
            results = find_opt_hyperparams(problem_class=None, run_nn=True, algorithm='genetic_alg', x_train=x_train,
                                           y_train=y_train, x_test=x_test, y_test=y_test)
            print(results)

        time, score, fitness_curve = run_neural_net("GA", x_train, y_train, x_test, y_test, 'genetic_alg')
        times[3] = time
        if plot_loss_curve:
            print("GA: ", fitness_curve.shape)
            plot_loss('genetic_alg', fitness_curve[:, 0])

    if plot_times:
        plot_nn_run_times(times, arange(4), algs)


if __name__ == "__main__":
    main()
