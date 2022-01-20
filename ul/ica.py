from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV

class ICA:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "ICA"
        self.name_abr = "ICA"
        self.fig_prefix = "ica_"
        self.base = FastICA(random_state=random_state, )

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
