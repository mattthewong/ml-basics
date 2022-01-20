from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection


class RandomizedProjections:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "Randomized Projections"
        self.name_abr = "RP"
        self.fig_prefix = "rp_"
        self.base = SparseRandomProjection(random_state=random_state)

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
