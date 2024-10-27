from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from modelBuilder import *
import tensorflow as tf
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from tensorflow import keras
from tensorflow.keras import layers
import modelWrapper

keras.utils.set_random_seed(812)
# standard random state
rand = 42

json_filepath = r"../res/combined_data/combined_cache.json"
target = ['EArrNom', 'GIncLss', 'TempLss', 'ModQual', 'OhmLoss', 'EArrMPP',
          'EArray', 'EffSysR', 'EffArrR']

# construct dataframe for model building
filepath = (
    r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch '
    r'Simulation\MOD_Canopy_Batch_Simulation_Project_BatchResults_combined.xlsx')
model_builder = ModelBuilder(filepath, target,
                             columns=["Indent", "Sheds Tilt", "Sheds Azim", "Comment", "EArray", "Syst_ON", "EArrNom",
                                      "GIncLss", "TempLss", "ModQual", "OhmLoss", "MisLoss", "EArrMPP", "EffArrR",
                                      "EffSysR", "EffSysC"], data_name="combined_data")

# build models
model_params = {}

''' PARAMETER DICTIONARIES FOR MODELS'''
# define dictionaries
dtree_params = {'random_state': rand}
dtree_params_cv = {'random_state': [rand], 'max_depth': [5, 10, 15, 20, 50, None], 'min_samples_split': [2, 3, 4, 5, 8],
                   'min_samples_leaf': [1, 2, 3]}

multi_svr_params = {}
multi_svr_params_cv = {'estimator__kernel': ['rbf'],
                       'estimator__C': [0.1, 1], 'estimator__epsilon': [0.001, 0.01, 0.1]}

nn_params = {'random_state': rand, 'activation': 'relu', 'alpha': 10, 'hidden_layer_sizes': [18, 24, 18],
             'learning_rate': 'constant', 'learning_rate_init': 0.1, 'solver': 'adam'}
nn_params_cv = {'random_state': [rand], 'batch_size': [64, 200], 'shuffle': [True, False],
                'early_stopping': [True],
                'alpha': [.0001, 10],
                'learning_rate_init': [0.0001, 0.1],
                'hidden_layer_sizes': [[18, 24, 18], [128, 256, 128]]}

seq_nn_params = {'layers': [layers.Dense(32, activation='relu'),
                            layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                            layers.Dense(64, activation='relu', kernel_regularizer='l2'),
                            layers.Dropout(0.1),
                            layers.Dense(32, activation='relu', kernel_regularizer='l2'),
                            layers.Dense(10)]}

# instantiate models
multi_svr_model = MultiOutputRegressor(SVR())
tuned_multi_svr_model = MultiOutputRegressor(SVR())

dtree_model = DecisionTreeRegressor(**dtree_params)
tuned_dtree_model = DecisionTreeRegressor(**dtree_params_cv)

nn_model = MLPRegressor(**nn_params)
tuned_nn_model = MLPRegressor(**nn_params_cv)

seq_nn_model = keras.Sequential(**seq_nn_params)
callback = keras.callbacks.EarlyStopping(patience=4)

# add to parameter dictionary
model_params[str(dtree_model.__class__.__name__)] = {"model": dtree_model, "param_grid": dtree_params}
model_params[str(tuned_dtree_model.__class__.__name__) + "_tuned"] = {"model": tuned_dtree_model,
                                                                      "param_grid": dtree_params_cv}

model_params[str(multi_svr_model.__class__.__name__)] = {"model": multi_svr_model, "param_grid": multi_svr_params}
model_params[str(tuned_multi_svr_model.__class__.__name__) + "_tuned"] = {"model": tuned_multi_svr_model,
                                                                          "param_grid": multi_svr_params_cv}

model_params[str(nn_model.__class__.__name__)] = {"model": nn_model, "param_grid": nn_params}
model_params[str(tuned_nn_model.__class__.__name__) + "_tuned"] = {"model": tuned_nn_model, "param_grid": nn_params_cv}

layer_configs = []
for layer in seq_nn_params.get('layers'):
    layer_config = layer.get_config()
    configs = {'name': layer_config.get('name'), 'units': layer_config.get('units'),
               'activation': layer_config.get('activation'),
               'kernel_regularization': layer_config.get('kernel_regularization'), 'rate': layer_config.get('rate')}
    layer_configs.append(configs)

model_params[str(seq_nn_model.__class__.__name__)] = {"model": seq_nn_model, "param_grid": {'layers': layer_configs,
                                                                                            "compile_params": {
                                                                                                'loss': 'mean_absolute_error',
                                                                                                'optimizer': tf.keras.optimizers.Adam(
                                                                                                    0.0001)},
                                                                                            "fit_params": {
                                                                                                'validation_split': 0.2,
                                                                                                'verbose': 2,
                                                                                                'epochs': 100,
                                                                                                'callbacks': [callback],
                                                                                                'batch_size': 16}}}

model_builder.run_model_builder(model_params, json_filepath)
