"""
This module contains the DataSplitter class, which is responsible for splitting the dataset into training and testing sets.
"""

import pandas as pd

class DataSplitter:
    """
    DataSplitter class to split the dataset into training and testing sets.
    """

    def __init__(self, config: dict):
        """
        Initializes the DataSplitter with the configuration dictionary.

        :param config: Configuration dictionary containing split parameters.
        """
        self.config = config

    def loocv_corpus_split(self, X: pd.DataFrame, wer: pd.Series, corpus: pd.Series) -> dict:
        """
        Splits the data into training and testing sets using Leave-One-Out Cross-Validation (LOOCV).

        :param X: Features DataFrame.
        :param wer: Word Error Rate Series.
        :param corpus: Corpus identifiers Series.
        :return: Dictionary containing training and testing data for each corpus.
        """
        splitted_data = {}
        
        unique_corpora = corpus.unique()
        
        for corp in unique_corpora:
            test_mask = corpus == corp
            train_mask = ~test_mask
            
            X_train = X[train_mask]
            wer_train = wer[train_mask]
            
            X_test = X[test_mask]
            wer_test = wer[test_mask]
            
            splitted_data[corp] = {
                'X_train': X_train,
                'wer_train': wer_train,
                'X_test': X_test,
                'wer_test': wer_test
            }
        
        return splitted_data
    
