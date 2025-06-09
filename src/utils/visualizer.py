import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class PreprocessingVisualizer:
    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def plot_feature_histograms(self, X, feature_names=None, title_prefix="", bins=30):
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        n_features = X.shape[1]
        n_cols = min(4, n_features)
        n_rows = int(np.ceil(n_features / n_cols))
        plt.figure(figsize=(n_cols*4, n_rows*3))
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i+1)
            plt.hist(X[:, i], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title(f"{title_prefix}{feature_names[i]}")
            plt.xlabel(feature_names[i])
        plt.tight_layout()
        plt.show(block=False)

    def plot_feature_boxplots(self, X, feature_names=None, title="Feature Boxplots"):
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        plt.figure(figsize=(max(10, X.shape[1]*0.7), 6))
        plt.boxplot(X, labels=feature_names, vert=True, patch_artist=True)
        plt.title(title)
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_corr_heatmap(self, X, feature_names=None, title="Feature Correlation Heatmap"):
        feature_names = feature_names or self.feature_names or [f"Feature {i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        corr = df.corr()
        plt.figure(figsize=(max(8, len(feature_names)), max(6, len(feature_names))))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title(title)
        plt.show()