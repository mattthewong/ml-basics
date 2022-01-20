from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


class NeuralNet:
    def __init__(self,
                 verbose=False,
                 alpha=1e-5,
                 hidden_layer_sizes=(10, 2),
                 random_state=1,
                 param_grid={},
                 ):
        self.name = "Neural Network"
        self.name_abr = "NN"
        self.fig_prefix = "nn_"
        self.base = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=1500,
            early_stopping=True,
            alpha=alpha,
            random_state=random_state,
            verbose=verbose,
        )
        self.clf = GridSearchCV(
            estimator=self.base,
            param_grid=param_grid,
            return_train_score=True,
            cv=5,
            n_jobs=-1,
            refit=True
        )
        self.verbose = verbose
