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
from sklearn.model_selection import GridSearchCV


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
        print("INVOKING...")
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