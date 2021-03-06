import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pipeline

########## Define Directories
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent # define actual path
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'

########## Define Target Variable from data source
TARGET = 'SalePrice'

########## Define features variables from data source
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood', 'OverallQual',
            'OverallCond', 'YearRemodAdd', 'RoofStyle', 'MasVnrType',
            'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
            '1stFlrSF', 'GrLivArea', 'BsmtFullBath', 'KitchenQual',
            'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageFinish',
            'GarageCars', 'PavedDrive', 'LotFrontage',
            # this variable is only to calculate temporal variable:
            'YrSold']

########## Skeleton of functions
def save_pipeline(*, pipeline_to_persist) -> None:
    """Persist the pipeline."""

    save_file_name = 'regression_model.pk1'
    save_path = TRAINED_MODEL_DIR / save_file_name
    joblib.dump(pipeline_to_persist, save_path)

    print('Saved Pipeline')

def run_training() -> None:
    """Train the model."""
    print('Training...')

    # read training data
    data = pd.read_csv(TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size = 0.1,
        random_state = 0) # set seed here

    # transform the target
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # when i call "fit" i call the entire pipeline to run
    # the entire pipeline calls the function on "preprocessors" and save the model
    pipeline.price_pipe.fit(X_train[FEATURES], y_train)

    save_pipeline(pipeline_to_persist = pipeline.price_pipe)


if __name__ == '__main__':
    run_training()
