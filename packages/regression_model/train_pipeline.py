import pathlib

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
def save_pipeline() -> None:
    """Persist the pipeline."""

    pass


def run_training() -> None:
    """Train the model."""

    print('Training...')


if __name__ == '__main__':
    run_training()
