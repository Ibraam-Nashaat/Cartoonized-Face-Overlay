import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.alphas = []
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        w = np.ones(n_samples)
        w[y == 1] *= 1 / (2 * n_pos)
        w[y == 0] *= 1 / (2 * n_neg)

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)

            y_pred = model.predict(X)
            err = np.sum(w * (y_pred != y))

            alpha = 0.5 * np.log((1 - err) / max(err, 1e-10))
            w = w * np.exp(-alpha * y_pred * y)
            w /= np.sum(w)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            print(model.predict(X))
            predictions += alpha * model.predict(X)

        return np.sign(predictions)
    
    def predict_th(self, X, th):
        predictions = np.zeros(X.shape[0])

        for model, alpha in zip(self.models, self.alphas):
            h = model.predict(X)
            h[h == 0] = -1
            predictions += alpha * model.predict(X)
        
        predictions[predictions >= th] = 1
        predictions[predictions < th] = 0

        return predictions