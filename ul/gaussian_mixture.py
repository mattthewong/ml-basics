from sklearn.mixture import GaussianMixture as gauss_mix
from sklearn.model_selection import GridSearchCV


class GMM:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "Gaussian Mixture"
        self.name_abr = "Gaussian Mixture"
        self.fig_prefix = "gaussian_mixture"
        self.base = gauss_mix

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
