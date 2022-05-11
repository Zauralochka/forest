# import logging
import numpy as np
​
from pathlib import Path
from typing import Tuple
​
import click
import pandas as pd
from sklearn.model_selection import train_test_split
​
​
def get_data(
    csv_path: Path,
    random_state: int,
    test_split_ratio: float = 0.3,
    target: str = 'Cover_Type'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    dataset = pd.read_csv(csv_path, index_col='Id')
    click.echo(f"Dataset shape: {dataset.shape}.")
    features_train, features_valid, target_train, target_valid = train_test_split(
        dataset.drop(target, axis=1), dataset[target],
        test_size=test_split_ratio, random_state=random_state
    )
    return features_train, features_valid, target_train, target_valid
​
def create_submission(predicted_labels, out_file, target='Cover_Type',
                      index_label="Id", init_index=15121):
    # Turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(
        predicted_labels,
        index = np.arange(
            init_index, predicted_labels.shape[0] + init_index
        ),
        columns=[target]
    )
    predicted_df.to_csv(out_file, index_label=index_label)