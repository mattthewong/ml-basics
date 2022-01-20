from sklearn import svm
from sklearn.model_selection import GridSearchCV


class SVMLearner:
    def __init__(self, kernel='linear', param_grid={}, verbose=False):
        self.name = "SVM " + kernel
        self.name_abr = "SVM_" + kernel
        self.fig_prefix = "svm_" + kernel
        self.base = svm.SVC(
            kernel=kernel,
            max_iter=1000
        )
        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=4,
            n_jobs=-1,
            refit=True
        )
        self.verbose = verbose