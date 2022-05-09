import logging
from pathlib import Path
from typing import Tuple

import click
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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