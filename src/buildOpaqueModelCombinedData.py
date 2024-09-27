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

# export model


# construct dataframe for model building
filepath = r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\combined_canopy_data.xlsx'
model_df = construct_df(filepath, columns=["Sheds Tilt", "Sheds Azim", "EArray (KWh)"])

# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)
pv_type = filepath.split('\\')[-1].replace(".xlsx", "")
export_model(scaler, '../res/combined_canopy_data/', False)

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)

# create data structures to capture model results
model_names = []
rmse_list = []
si_list = []

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
nn_params_cv = {'random_state': [rand], 'solver': ('sgd', 'adam'),
                'alpha': [.0001, .001, .01, 1],
                'learning_rate': ('constant', 'adaptive'),
                'learning_rate_init': [0.0001, .001, 0.01],
                'hidden_layer_sizes': [[50, 100, 50]]}

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

# run models that haven't previously been run
for model in model_params.keys():
    curr_model = model_params.get(model).get("model")
    curr_param = model_params.get(model).get("param_grid")
    skip_model = check_models_to_run(curr_model, curr_param, pv_type)
    if not skip_model:
        if 'tuned' in model:
            train_model, rmse_score, si_score, model_names, rmse_list, si_list, best_params = tune_models(curr_model,
                                                                                                          curr_param,
                                                                                                          X_train,
                                                                                                          y_train,
                                                                                                          X_test,
                                                                                                          y_test,
                                                                                                          model_names,
                                                                                                          rmse_list,
                                                                                                          si_list,
                                                                                                          verbose=5)
            score = "_" + str(round(rmse_score, 2))
            export = add_model_params(pv_type, curr_param, model, rmse_score, si_score, best_params)
            # only export tuned model if it is best tuned model
            if export:
                export_model(train_model, pv_type, True, score)

        else:
            if model_params.get(model).get("fit_params"):
                compile_params = model_params.get(model).get("compile_params")
                fit_params = model_params.get(model).get("fit_params")
                train_model, rmse_score, si_score, model_names, rmse_list, si_list = build_tf_models(curr_model,
                                                                                                     X_train,
                                                                                                     y_train,
                                                                                                     X_test, y_test,
                                                                                                     model_names,
                                                                                                     rmse_list, si_list,
                                                                                                     compile_params,
                                                                                                     fit_params)
                callback_params = model_params.get(model).get("fit_params").get("callbacks")[0]
                fit_params['callbacks'] = {'monitor': callback_params.monitor,
                                           'patience': callback_params.patience}
                score = "_" + str(round(rmse_score, 2))
                layer_configs = []
                for layer in seq_nn_params:
                    layer_config = layer.get_config()
                    configs = {'units': layer_config['units'], 'activation': layer_config['activation']}
                    layer_configs.append(configs)
                all_params = {"layers": layer_configs}
                serial_compile_params = {'loss': compile_params.get('loss'),
                                         'optimizer': [compile_params.get('optimizer').get_config()['name'],
                                                       compile_params.get('optimizer').get_config()['learning_rate'],
                                                       compile_params.get('optimizer').get_config()['epsilon']]}
                all_params.update(serial_compile_params)
                all_params.update(fit_params)
                add_model_params(pv_type, all_params, model, rmse_score, si_score)

            else:
                train_model, rmse_score, si_score, model_names, rmse_list, si_list = build_models(curr_model, X_train,
                                                                                                  y_train,
                                                                                                  X_test, y_test,
                                                                                                  model_names,
                                                                                                  rmse_list, si_list)
                score = "_" + str(round(rmse_score, 2))
                add_model_params(pv_type, curr_param, model, rmse_score, si_score)

            export_model(train_model, pv_type, False, score)
export_to_csv(pv_type)
