from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from jsonWriter import *
from modelBuilder import *
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(812)
# standard random state
rand = 42

json_filepath = r"../res/ohmloss_data/ohmloss_cache.json"
target = "OhmLoss (KWh)"

# construct dataframe for model building
filepath = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\ohmloss_data.xlsx'
model_builder = ModelBuilder(filepath, target,
                             columns=["Sheds Tilt", "Sheds Azim", target])


# build models
model_params = {}

''' PARAMETER DICTIONARIES FOR MODELS'''
# define dictionaries
svr_params = {}
svr_params_cv = {'kernel': ['rbf'],
                 'C': [0.1, 1], 'epsilon': [0.1, 0.2, 0.5]}

ridge_params = {}
ridge_params_cv = {'solver': ('auto', 'svd', 'lsqr'),
                   'alpha': [1, 1.5, 5, 10, 20, 50]}

lasso_params = {}
lasso_params_cv = {'alpha': [0.01, 0.1, 0.5, 1, 1.5, 5, 10, 20, 50]}

nn_params = {'random_state': rand, 'activation': 'relu', 'alpha': 10, 'hidden_layer_sizes': [18, 24, 18],
             'learning_rate': 'constant', 'learning_rate_init': 0.1, 'solver': 'adam'}
nn_params_cv = {'random_state': [rand], 'batch_size': [64, 200], 'shuffle': [True, False],
                'early_stopping': [True],
                'alpha': [.0001, 10],
                'learning_rate_init': [0.0001, 0.1],
                'hidden_layer_sizes': [[18, 24, 18], [128, 256, 128]]}

seq_nn_params = [layers.Dense(32, activation='relu'),
                 layers.Dense(64, activation='relu'),
                 layers.Dense(64, activation='relu'),
                 layers.Dense(32, activation='relu'),
                 layers.Dense(1)]

# instantiate models
svr_model = SVR(**svr_params)
tuned_svr_model = SVR(**svr_params_cv)

ridge_model = Ridge(**ridge_params)
tuned_ridge_model = Ridge(**ridge_params_cv)

lasso_model = linear_model.Lasso(**lasso_params)
tuned_lasso_model = linear_model.Lasso(**lasso_params_cv)

nn_model = MLPRegressor(**nn_params)
tuned_nn_model = MLPRegressor(**nn_params_cv)

seq_nn_model = keras.Sequential(seq_nn_params)
callback = keras.callbacks.EarlyStopping(patience=4)

# add to parameter dictionary
model_params[str(svr_model.__class__.__name__)] = {"model": svr_model, "param_grid": svr_params}
model_params[str(tuned_svr_model.__class__.__name__) + "_tuned"] = {"model": tuned_svr_model,
                                                                    "param_grid": svr_params_cv}

model_params[str(ridge_model.__class__.__name__)] = {"model": ridge_model, "param_grid": ridge_params}
model_params[str(tuned_ridge_model.__class__.__name__) + "_tuned"] = {"model": tuned_ridge_model,
                                                                      "param_grid": ridge_params_cv}

model_params[str(lasso_model.__class__.__name__)] = {"model": lasso_model, "param_grid": lasso_params}
model_params[str(tuned_lasso_model.__class__.__name__) + "_tuned"] = {"model": tuned_lasso_model,
                                                                      "param_grid": lasso_params_cv}

model_params[str(nn_model.__class__.__name__)] = {"model": nn_model, "param_grid": nn_params}
model_params[str(tuned_nn_model.__class__.__name__) + "_tuned"] = {"model": tuned_nn_model, "param_grid": nn_params_cv}

model_params[str(seq_nn_model.__class__.__name__)] = {"model": seq_nn_model, "param_grid": seq_nn_params,
                                                      "compile_params": {'loss': 'mean_absolute_error',
                                                                         'optimizer': tf.keras.optimizers.Adam(0.0001)},
                                                      "fit_params": {'validation_split': 0.2, 'verbose': 2,
                                                                     'epochs': 100, 'callbacks': [callback]}}

model_builder.run_model_builder(model_params, "ohmloss_data", json_filepath)