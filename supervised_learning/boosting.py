from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

class BoostingDT:
    def __init__(self, param_grid={}, random_state=1, verbose=False):
        self.name = "AdaBoost"
        self.name_abr = "AdaBoost"
        self.fig_prefix = "adaboost_"
        self.tree = tree.DecisionTreeClassifier(max_depth=2, min_samples_leaf=1, random_state=random_state)
        self.base = AdaBoostClassifier(
            base_estimator=self.tree,
        )

        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            # verbose=10,
            n_jobs=-1,
            refit=True
        )

        self.verbose = verbose
