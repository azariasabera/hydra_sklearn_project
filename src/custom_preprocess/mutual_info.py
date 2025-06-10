from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression
import numpy as np

class MutualInformation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1):
        self.threshold = threshold
        self.feature_indices = None

    def fit(self, X, y):
        mi = mutual_info_regression(X, y)
        self.feature_indices = np.where(mi > self.threshold)[0]
        #print(self.feature_indices)
        return self 

    def transform(self, X):
        # Support DataFrame and numpy array
        if hasattr(X, "iloc"):
            return X.iloc[:, self.feature_indices]
        else:
            return X[:, self.feature_indices]