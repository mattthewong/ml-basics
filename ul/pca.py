from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV


class PCA_Class:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "PCA"
        self.name_abr = "PCA"
        self.fig_prefix = "pca_"
        self.base = PCA(random_state=random_state)

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
