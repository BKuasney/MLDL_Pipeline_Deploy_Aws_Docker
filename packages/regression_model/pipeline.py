from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import preprocessors as pp

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual',
                            'BsmtExposure', 'FireplaceQu',
                            'GarageType', 'GarageFinish']

TEMPORAL_VARS = 'YearRemodAdd'

# this variable is to calculate the temporal variable,
# can be dropped afterwards
DROP_FEATURES = 'YrSold'

# variables to log transform
NUMERICAL_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['LotFrontage']


########### Define preprocessing of data
# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning',
                    'Neighborhood',
                    'RoofStyle',
                    'MasVnrType',
                    'BsmtQual',
                    'BsmtExposure',
                    'HeatingQC',
                    'CentralAir',
                    'KitchenQual',
                    'FireplaceQu',
                    'GarageType',
                    'GarageFinish',
                    'PavedDrive']

PIPELINE_NAME = 'lasso_regression'

# Call functions from a package and runs "step-by-step"
# each of them is a step of transformation
price_pipe = Pipeline(
    [
        ('categorical_imputer', pp.CategoricalImputer(variables=CATEGORICAL_VARS)),
        ('numerical_imputer', pp.NumericalImputer(variables=NUMERICAL_VARS_WITH_NA)),
        ('temporal_variables', pp.TemporalVariableEstimator(variables=TEMPORAL_VARS, reference_variable=TEMPORAL_VARS)),
        ('rare_label_encoder', pp.RareLabelCategoricalEncoder(tol=0.01, variables=CATEGORICAL_VARS)),
        ('categorical_encoder', pp.CategoricalEncoder(variables=CATEGORICAL_VARS)),
        ('log_transformer', pp.LogTransformer(variables=NUMERICAL_LOG_VARS)),
        ('drop_features', pp.DropUnecessaryFeatures(variables_to_drop=DROP_FEATURES)),
        ('scaler', MinMaxScaler()),
        ('linear_model', Lasso(alpha=0.005, random_state=0))
    ])
