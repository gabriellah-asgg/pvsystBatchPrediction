from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from jsonWriter import *
from modelBuilder import *

rand = 42

# build models
model_params = {}

''' PARAMETER DICTIONARIES FOR MODELS'''
# define dictionaries
svr_params = {}
svr_params_cv = {'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
                 'C': [1, 10]}

ridge_params = {}
ridge_params_cv = {'solver': ('auto', 'svd', 'lsqr'),
                   'alpha': [1, 1.5, 5, 10, 20, 50]}

lasso_params = {}
lasso_params_cv = {'alpha': [0.01, 0.1, 0.5, 1, 1.5, 5, 10, 20, 50]}

nn_params = {'random_state': rand, 'activation': 'relu', 'alpha': 10, 'hidden_layer_sizes': [18, 24, 18],
             'learning_rate': 'constant', 'learning_rate_init': 0.1, 'solver': 'adam'}
nn_params_cv = {'solver': ('sgd', 'adam'), 'alpha': [.0001, .001, .01, 1],
                'learning_rate': ('constant', 'adaptive'),
                'learning_rate_init': [.001, .01, .1],
                'hidden_layer_sizes': [[18, 24, 18], [20, 50, 20]]}

filepath = r"Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch Simulation_0815\Semi Opaque Panels_Worst Case Scenario.CSV"
columns = ["Ident", "Sheds Tilt", "Sheds Azim", "NB Strings In Parall", "Comment", "Error", "EArray (KWh)", "LCR Ratio",
           "Ls", "Lc"]
model_df = construct_df(filepath, columns)

# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)
pv_type = filepath.split('\\')[-1].replace(".CSV", "")
export_model(scaler, pv_type, False)

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)

# create data structures to capture model results
model_names = []
rmse_list = []
si_list = []

# instantiate models
svr_model = SVR(**svr_params)
tuned_svr_model = SVR(**svr_params_cv)

ridge_model = Ridge(**ridge_params)
tuned_ridge_model = Ridge(**ridge_params_cv)

lasso_model = linear_model.Lasso(**lasso_params)
tuned_lasso_model = linear_model.Lasso(**lasso_params_cv)

nn_model = MLPRegressor(**nn_params)
tuned_nn_model = MLPRegressor(**nn_params_cv)

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
            train_model, rmse_score, si_score, model_names, rmse_list, si_list = build_models(curr_model, X_train,
                                                                                              y_train, X_test, y_test,
                                                                                              model_names, rmse_list,
                                                                                              si_list)
            score = "_" + str(round(rmse_score, 2))
            export_model(train_model, pv_type, False, score)
            add_model_params(pv_type, curr_param, model, rmse_score, si_score)
