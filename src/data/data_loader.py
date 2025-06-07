"""
This module handles the data loading, preparing a dataframe from a CSV file
"""

import pandas as pd
from omegaconf import OmegaConf

class DataLoader:
    """
    DataLoader class to load and prepare data from a CSV file.
    """

    def __init__(self, config: dict, csv_file: str):
        """
        Initializes the DataLoader with the path to the CSV file.

        :param csv_file: Path to the CSV file containing the data.
        """
        self.config = config
        self.csv_file = csv_file

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file into a pandas DataFrame.

        :return: A pandas DataFrame containing the loaded data.
        """

        # in config all data types are defined as strings, omegaConf safely converts them to dict
        dtype_map = OmegaConf.to_container(self.config.data_types, resolve=True)

        df = pd.read_csv(
            self.csv_file,
            sep="\t",                # Use tab as the separator
            header=0,                # First row is the header
            dtype=dtype_map,  # Data types for each column
            skip_blank_lines=False,  # Do not skip blank lines
            engine="python"          # Use Python engine for better compatibility
        )

        # Replace NaN values in specific columns with empty strings
        for col in ["file", "ref_text", "text", "text_small", "ds"]:
            if col in df.columns:
                df[col] = df[col].replace(pd.NA, "")

        if self.config.params.include_opensmile:
            # Load opensmile features if specified in the config
            opensmile_features_path = self.config.paths.opensmile_path
            if not opensmile_features_path: 
                opensmile_df = pd.read_csv(opensmile_features_path, sep="\t")
                df = pd.merge(df, opensmile_df, on='file', how='left') # This assumes 'file' is the common column and number of rows match

        return df
    
    def extract_data(self, df: pd.DataFrame) -> tuple:
        """
        Extracts features, word error rates, and corpus identifiers from the DataFrame.

        :param df: The DataFrame containing the data.
        :return: A tuple containing features (X), word error rates (wer), and corpus identifiers.
        """

        try:
            wer = df['wer'].copy()
            corpus = df['ds'].copy()
            X = df[self.config.features.main_features].copy()
        except KeyError as e:
            raise KeyError(f"Missing required column in DataFrame: {e}")

        # Remove rows with NaN values in the features
        X = X.dropna()

        # Remove rows where duration is less than the threshold
        if 'duration' in df.columns:
            X = X[X['duration'] >= self.config.params.duration_threshold]
            wer = wer[X.index]
            corpus = corpus[X.index]

        return X, wer, corpus