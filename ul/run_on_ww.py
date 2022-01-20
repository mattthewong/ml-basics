import sys
from numpy import arange, zeros, linspace, logspace, nan
from utils import get_data, \
    determine_train_and_query_times, plot_training_times, plot_query_times, \
    gen_and_plot_validation_curve, fit_base, gen_and_plot_learning_curve, \
    generate_silhoutte_score_plot, generate_kmeans_sil_icd_plots
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from k_means import K_Means
from gaussian_mixture import GMM
from pca import PCA_Class
from ica import ICA
from svd import SVDT
from rp import RandomizedProjections
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from neural_net_sci import NeuralNet
import matplotlib.pyplot as plt

# NN work: 3 learner curves per dim red alg, so 12 learning curves per dataset.
# 20 min per learning curve = ~4 hours per dataset

def pairwise_dist_corr(x1, x2):
    assert x1.shape[0] == x2.shape[0]

    d1 = pairwise_distances(x1)
    d2 = pairwise_distances(x2)
    return np.corrcoef(d1.ravel(), d2.ravel())[0, 1]


def run_kmeans(x_train, learner, file):
    generate_silhoutte_score_plot(x_train, 60, learner, file)
    k = list(range(2, 8))
    generate_kmeans_sil_icd_plots(x_train, k, learner, file)


def run_em(x_train, learner, file):
    generate_silhoutte_score_plot(x_train, 60, learner, file)


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
    plot_vc = False  # plot validation curve
    plot_sil = False  # plot silhouette scores
    plot_kmeans_icd = False  # plot intercluster distance plots
    plot_em = False
    generate_drs = False
    plot_times = False  # plot train and query times
    plot_var = True
    run_pca = True
    run_ica = True
    run_rp = True
    run_svd = True
    run_nn = True
    run_nn_cluster = False
    verbose = False  # print debug statements

    # init time structs
    dim_red_algs = ('PCA-KM', 'PCA-GM', 'ICA-KM', 'ICA-GM', 'RP-KM', 'RP-GM', 'SVD-KM', 'SVD-GM')
    train_times = zeros(2)
    query_times = zeros(2)
    train_times_nn = zeros(8)
    query_times_nn = zeros(8)

    pg_nn = {'alpha': [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 2, 3],
             'learning_rate_init': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5, 0.8, 1],
             'hidden_layer_sizes': [(10, 10), (100, 100)]}

    # K Means
    pg = {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]}
    learner = K_Means(verbose=verbose, param_grid=pg)
    learner.base(8).fit(x_train)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "# of Clusters", "n_clusters", pg['n_clusters'])
    if plot_sil:
        generate_silhoutte_score_plot(x_train, 60, learner, file)
    if plot_kmeans_icd:
        k = list(range(2, 8))
        generate_kmeans_sil_icd_plots(x_train, k, learner, file)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[0] = t_time
        query_times[0] = q_time

    # Gaussian Mixture
    pg = {'n_components': [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]}
    learner = GMM(verbose=verbose, param_grid=pg)
    learner.base(8).fit(x_train)
    if plot_vc:
        gen_and_plot_validation_curve(learner, file, x_train, y_train, "# of Components", "n_components",
                                      pg['n_components'])
    if plot_sil:
        generate_silhoutte_score_plot(x_train, 60, learner, file)
    if plot_lc:
        gen_and_plot_learning_curve(learner, file, x_train, y_train)
    if plot_times:
        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
        train_times[1] = t_time
        query_times[1] = q_time

    # plot times
    if plot_times:
        # plot times
        plot_training_times(train_times, arange(2), classifiers)
        plot_query_times(query_times, arange(2), classifiers)

    # -- Dimensionality Reduction --

    # PCA
    if run_pca:
        if plot_var:
            learner = PCA_Class(verbose=verbose)
            pca = learner.base.fit(x)
            plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
            plt.title("Component-wise Explained Variance")
            plt.xlabel('# of Components')
            plt.ylabel('Explained variance')
            plt.title("PCA Variance")
            plt.savefig("variance_pca_stroke.png")
            plt.clf()

        if generate_drs:
            for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                learner = PCA_Class(verbose=verbose)
                learner.base.set_params(n_components=dim)
                x_pca = pd.DataFrame(learner.base.fit_transform(x))
                print(x_pca.shape, y.shape)
                if plot_kmeans_icd:
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_pca)
                    run_kmeans(x_pca, learner, file + "_dim_red_%d_pca" % dim)
                if plot_em:
                    learner = GMM(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_pca)
                    run_em(x_pca, learner, file + "_dim_red_%d_pca" % dim)
                # NN
                if run_nn and dim == 2:
                    print("Running neural net on PCA dim red dataset...")
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_pca" % dim, x_train, y_train)
                if run_nn_cluster and dim == 2:
                    # K Means first
                    print(f'Running neural net on PCA dim red data with kmeans features...')
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_pca)
                    y_pred = base.predict(x_pca)
                    result = pd.concat([pd.DataFrame(x_pca), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_pca_kmeans_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[pca_km]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[0] = t_time
                        query_times_nn[0] = q_time
                    # EM
                    print(f'Running neural net on PCA dim red data with em features...')
                    learner = GMM(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_pca)
                    y_pred = base.predict(x_pca)
                    result = pd.concat([pd.DataFrame(x_pca), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_pca_gmm_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[pca_gmm]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[1] = t_time
                        query_times_nn[1] = q_time

    # ICA
    if run_ica:
        if plot_var:
            learner = ICA(verbose=verbose)
            learner.base.fit(x_train)
            kurt = []
            for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                learner.base.set_params(n_components=dim)
                tmp = learner.base.fit_transform(x_train)
                tmp = pd.DataFrame(tmp)
                tmp = tmp.kurt(axis=0)
                kurt.append(tmp.abs().mean())
            plt.plot(range(1, 10), kurt)
            plt.title("Component-wise Kurtosis")
            plt.xlabel('# of Components')
            plt.ylabel('Kurtosis')
            plt.title("ICA Kurtosis")
            plt.savefig("variance_ica_stroke.png")
            plt.clf()

        if generate_drs:
            for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                learner = ICA(verbose=verbose)
                learner.base.set_params(n_components=dim)
                x_ica = pd.DataFrame(learner.base.fit_transform(x))
                if plot_kmeans_icd:
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_ica)
                    run_kmeans(x_ica, learner, file + "_dim_red_%d_ica" % dim)
                if plot_em:
                    learner = GMM(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_ica)
                    run_em(x_ica, learner, file + "_dim_red_%d_ica" % dim)
                # NN
                if run_nn and dim == 2:
                    print("Running neural net on ICA dim red dataset...")
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(x_ica, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_ica" % dim, x_train, y_train)
                if run_nn_cluster and dim == 2:
                    # K Means first
                    print(f'Running neural net on ICA dim red data with kmeans features...')
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_ica)
                    y_pred = base.predict(x_ica)
                    result = pd.concat([pd.DataFrame(x_ica), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_ica_kmeans_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[ica_km]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[2] = t_time
                        query_times_nn[2] = q_time
                    # EM
                    print(f'Running neural net on ICA dim red data with em features...')
                    learner = GMM(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_ica)
                    y_pred = base.predict(x_ica)
                    result = pd.concat([pd.DataFrame(x_ica), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_ica_gmm_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[ica_gmm]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[3] = t_time
                        query_times_nn[3] = q_time

    # RP
    if run_rp:
        if plot_var:
            learner = ICA(verbose=verbose)
            learner.base.fit(x_train)
            corr = []
            for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                learner = RandomizedProjections(verbose=verbose, random_state=rs)
                learner.base.set_params(n_components=dim)
                learner.base.fit(x_train)
                corr.append(pairwise_dist_corr(learner.base.fit_transform(x_train), x_train))
            plt.plot(range(1, 10), corr)
            plt.title("RP Component-wise Pairwise Distance Correlation")
            plt.xlabel('# of Components')
            plt.ylabel('PDC')
            plt.savefig("variance_rp_stroke.png")
            plt.clf()
        if generate_drs:
            for dim in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                learner = RandomizedProjections(verbose=verbose)
                learner.base.set_params(n_components=dim)
                x_rp = pd.DataFrame(learner.base.fit_transform(x))
                if plot_kmeans_icd:
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_rp)
                    run_kmeans(x_rp, learner, file + "_dim_red_%d_rp" % dim)
                if plot_em:
                    learner = GMM(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_rp)
                    run_em(x_rp, learner, file + "_dim_red_%d_rp" % dim)
                # NN
                if run_nn and (dim == 2 or dim == 6):
                    print("Running neural net on RP dim red dataset...")
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(x_rp, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_rp" % dim, x_train, y_train)
                if run_nn_cluster and dim == 2:
                    # K Means first
                    print(f'Running neural net on RP dim red data with kmeans features...')
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_rp)
                    y_pred = base.predict(x_rp)
                    result = pd.concat([pd.DataFrame(x_rp), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_rp_kmeans_cluster" % dim, x_train, y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[rp_km]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[4] = t_time
                        query_times_nn[4] = q_time
                    # EM
                    print(f'Running neural net on RP dim red data with em features...')
                    learner = GMM(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_rp)
                    y_pred = base.predict(x_rp)
                    result = pd.concat([pd.DataFrame(x_rp), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_rp_gmm_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[rp_gmm]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[5] = t_time
                        query_times_nn[5] = q_time

    # SVD
    if run_svd:
        if plot_var:
            learner = SVDT(random_state=rs)
            svd = learner.base.fit(x_train)
            print(svd.explained_variance_.shape)
            plt.plot(range(1, 10), svd.explained_variance_)
            plt.title("Component-wise Explained Variance")
            plt.xlabel('# of Components')
            plt.ylabel('Explained variance')
            plt.title("SVD Variance")
            plt.savefig("variance_svd_stroke.png")
            plt.clf()
        if generate_drs:
            for dim in [2, 3, 4, 5, 6, 7, 8, 9]:
                learner = SVDT(verbose=verbose)
                learner.base.set_params(n_components=dim)
                x_svd = pd.DataFrame(learner.base.fit_transform(x))
                if plot_kmeans_icd:
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_svd)
                    run_kmeans(x_svd, learner, file + "_dim_red_%d_svd" % dim)
                if plot_em:
                    learner = GMM(verbose=verbose, param_grid=pg)
                    learner.base(8).fit(x_svd)
                    run_em(x_svd, learner, file + "_dim_red_%d_svd" % dim)
                # NN
                if run_nn and dim == 2:
                    print("Running neural net on SVD dim red dataset...")
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(x_svd, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_svd" % dim, x_train, y_train)
                if run_nn_cluster and dim == 2:
                    # K Means first
                    print(f'Running neural net on SVD dim red data with kmeans features...')
                    learner = K_Means(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_svd)
                    y_pred = base.predict(x_svd)
                    result = pd.concat([pd.DataFrame(x_svd), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_svd_kmeans_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[svd_km]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[6] = t_time
                        query_times_nn[6] = q_time
                    # EM
                    print(f'Running neural net on SVD dim red data with em features...')
                    learner = GMM(verbose=verbose, param_grid=pg)
                    base = learner.base(2).fit(x_svd)
                    y_pred = base.predict(x_svd)
                    result = pd.concat([pd.DataFrame(x_svd), pd.DataFrame(y_pred)], axis=1, sort=False)
                    result.columns = [0, 1, 2]
                    learner = NeuralNet(verbose=verbose, random_state=1, param_grid=pg_nn)
                    x_train, x_test, y_train, y_test = train_test_split(result, y, test_size=0.4)
                    fit_base(learner, x_train, y_train, x_test, y_test)
                    gen_and_plot_learning_curve(learner, file + "_dim_red_%d_svd_gmm_cluster" % dim, x_train,
                                                y_train)
                    if plot_times:
                        t_time, q_time = determine_train_and_query_times(learner, x_train, y_train, x_test)
                        print(f'[svd_gmm]training time: {t_time}, q_time: {q_time}')
                        train_times_nn[7] = t_time
                        query_times_nn[7] = q_time
    # plot times
    if plot_times:
        # plot times
        plot_training_times(train_times_nn, arange(8), dim_red_algs)
        plot_query_times(query_times_nn, arange(8), dim_red_algs)

if __name__ == "__main__":
    main()
