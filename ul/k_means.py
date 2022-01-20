from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV


class K_Means:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "KMeans"
        self.name_abr = "KMeans"
        self.fig_prefix = "k_means_"
        self.base = KMeans

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
