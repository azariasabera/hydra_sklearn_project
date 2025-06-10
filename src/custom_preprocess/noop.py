"""A transformer that does nothing (no operation)."""

class NoOp:
    def fit(self, X_train, y=None):
        # Just return self, as fit does nothing
        return self

    def transform(self, X, y=None):
        # Return X unchanged
        return X

    def fit_transform(self, X_train, y=None):
        return X_train