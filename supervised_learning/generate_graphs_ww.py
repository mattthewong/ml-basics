import sys
from numpy import arange, zeros, linspace, logspace
from decision_tree import DTLearner
from boosting import BoostingDT
from knn import KNNLearner
from svm import SVMLearner
from neural_net_sci import NeuralNet
from utils import get_data, \
    determine_train_and_query_times, plot_training_times, plot_query_times, \
    gen_and_plot_validation_curve, fit_base, tune_hyperparameters, gen_and_plot_learning_curve
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    if len(sys.argv) != 1:
        print("Usage: python generate_graphs_ww.py")
        sys.exit(1)

    # get data
    x, y = get_data("winequality-white")

    x = preprocessing.scale(x)

    # seed
    rs = 51

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=rs)

    file = 'wine-quality-white'
    classifiers = ('Decision tree', 'AdaBoost', 'kNN', 'SVM', "Neural Network")

    # Control setting boolean flags
    plot_lc = True  # plot learning curve
    plot_vc = True  # plot validation curve
    plot_times = True  # plot train and query times
    verbose = True  # print debug statements

    # init time structs
    train_times = zeros(5)
    query_times = zeros(5)

    # Decision Tree
    pg = {'criterion': ['gini', 'entropy'], 'max_depth': arange(1, 21)}
    learner = DTLearner(verbose=verbose, random_state=rs, param_grid=pg)
    fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Min-Sample-Split", "min_samples_split",
                                      [0, 0.5, 1, 2, 3, 4])
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Max-Depth", "max_depth", arange(1, 21))
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)
    optimal_tree = learner.base
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[0] = t_time
        query_times[0] = q_time
    # # AdaBoost
    pg = {'n_estimators': arange(1, 100), 'learning_rate': linspace(0.01, 0.6, 10)}
    learner = BoostingDT(verbose=True, random_state=rs, param_grid=pg)
    learner.tree = optimal_tree
    learner.base = fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Learning Rate", "learning_rate",
                                      linspace(0.01, 0.6, 10))
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "N Estimators", "n_estimators", arange(1, 100))
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[1] = t_time
        query_times[1] = q_time
    # #
    # # # K-Nearest-Neighbors
    n_neighbors = 2
    pg = {'n_neighbors': arange(1, 50), 'weights': ['uniform', 'distance']}
    learner = KNNLearner(verbose=True, n_neighbors=n_neighbors, param_grid=pg)
    fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "K-Value", "n_neighbors", arange(1, 100))
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Power Metric", "p", [1, 2, 3, 4, 5])
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[2] = t_time
        query_times[2] = q_time

    # SVM Linear
    pg = {'C': logspace(-2, 1, 10), 'max_iter': logspace(1, 6, 4)}
    learner = SVMLearner(kernel='linear', verbose=True, param_grid=pg)
    fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "C", "C", logspace(-2, 1, 10))
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Max Iter", "max_iter", logspace(1, 6, 4))
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[3] = t_time
        query_times[3] = q_time

    # SVM Poly
    learner = SVMLearner(kernel='poly', verbose=True, param_grid=pg)
    fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "C", "C", logspace(-2, 1, 10))
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Max Iter", "max_iter", logspace(1, 6, 4))
    # skip plot times here.
    #
    # Neural Net
    pg = {
        'alpha': [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3],
        'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 1],
        'hidden_layer_sizes': [(10, 30, 10), (20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant', 'adaptive'],
    }
    learner = NeuralNet(verbose=True, random_state=1, param_grid=pg)
    fit_base(learner, x_train, y_train, x_test, y_test)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Alpha", "alpha",
                                      [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3])
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "Learning-Rate Init", "learning_rate_init",
                                      [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 1])
    tune_hyperparameters(learner, x_train, y_train, x_test, y_test)

    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
    train_times[4] = t_time
    query_times[4] = q_time

    # plot times
    if plot_times:
        # plot times
        plot_training_times(train_times, arange(5), classifiers)
        plot_query_times(query_times, arange(5), classifiers)


if __name__ == "__main__":
    main()
