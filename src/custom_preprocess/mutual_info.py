from sklearn.feature_selection import mutual_info_regression
import numpy as np

class MutualInformation:
    def __init__(self):
        self.feature_indices = []
    def fit(self, X_train, y_train):
        mi =  mutual_info_regression(X_train, y_train)
        self.feature_indices = np.where(mi > 0.1)[0]
        print(self.feature_indices)
        return X_train.iloc[:, self.feature_indices] # .iloc since X is a pandas df, y is pandas Series
    def transform(self, X_test):
        return X_test.iloc[:, self.feature_indices]