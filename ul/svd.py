from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD


class SVDT:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "SVD"
        self.name_abr = "SVD"
        self.fig_prefix = "singular value decomposition_"
        self.base = TruncatedSVD(random_state=random_state)

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
