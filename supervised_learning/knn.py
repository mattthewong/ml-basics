from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

class KNNLearner:
    def __init__(self, n_neighbors=2, param_grid={},verbose=False):
        self.name = "K-Nearest Neighbors"
        self.name_abr = "kNN"
        self.fig_prefix = "knn_"
        self.base = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )
        self.verbose = verbose