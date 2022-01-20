from sklearn import tree
from sklearn.model_selection import GridSearchCV

class DTLearner:
    def __init__(self, random_state=1, param_grid={}, verbose=False):
        self.name = "Decision Tree"
        self.name_abr = "DT"
        self.fig_prefix = "dt_"
        self.base = tree.DecisionTreeClassifier(random_state=random_state)
        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )
        self.verbose = verbose
