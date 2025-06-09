"""
Custom pipeline for the WER classifier.
"""

class CustomPipeline:
    def __init__(self, steps):
        self.steps = steps

    @property # to access model as a property
    def model(self):
        is_found = False
        for name, step in self.steps:
            if name == 'Model':
                is_found = True
                return step
        if not is_found:
            raise ValueError("step with name 'Model' not found")

    # wanted to call it fit_transform but to comply with the sklearn one and my implementation in build.py
    def fit(self, X_train, y_train):
        X, y = X_train, y_train
        for name, step in self.steps:
            # Model fitting is done after all preprocessing steps
            # ** Change this if statement since the name can be changed to anything **
            if name == 'Model':
                continue

            # Check if the step has fit_transform, fit and transform, or just transform methods
            if hasattr(step, 'fit_transform'):
                X = step.fit_transform(X, y)
            elif hasattr(step, 'fit') and hasattr(step, 'transform'):
                step.fit(X, y)
                X = step.transform(X)
            #elif hasattr(step, 'transform'):
            #    X = step.transform(X)
            else:
                raise ValueError(f"Step '{name}' does not have a fit_transform or fit-and-transform method(s).")
            
        # Finally fit the model
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        return X, y

    def transform(self, X_test):
        """
        Handles the transform of the testing data.

        Args:
            X_test (array-like): Testing features

        Returns:
            X (array-like): Transformed testing features
        """
        X = X_test
        for name, step in self.steps:
            if name == 'Model':
                continue
            if hasattr(step, 'transform'):
                X = step.transform(X)
            else:
                raise ValueError(f"Step '{name}' does not have a transform method.")
        return X
    
    def predict(self, X_test):
        """
        Predict using the pipeline.

        Args:
            X_test (array-like): Features for prediction.

        Returns:
            array-like: Predicted labels 1 or 0.
        """
        X = X_test
        # First transform the features
        X = self.transform(X)
            
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        else:
            raise ValueError("Model does not have a predict method.")

    def predict_proba(self, X_test):
        """
        Predict probabilities using the pipeline.

        Args:
            X_test (array-like): Features for prediction.

        Returns:
            array-like: Predicted probabilities.
        """
        X = X_test
        X = self.transform(X)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Model does not have a predict_proba method.") 