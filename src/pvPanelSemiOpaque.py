import pandas as pd

from preprocessData import Preprocessor
import preprocessData as preprocess
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from jsonWriter import check_models_to_run, add_model_params
import warnings

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

nn_params = {'random_state':rand, 'activation':'relu', 'alpha':10, 'hidden_layer_sizes':[18, 24, 18],
                  'learning_rate':'constant', 'learning_rate_init':0.1, 'solver':'adam'}
nn_params_cv = {'solver': ('sgd', 'adam'), 'alpha': [.0001, .001, .01, 1],
            'learning_rate': ('constant', 'adaptive'),
            'learning_rate_init': [.001, .01, .1], 'activation': 'relu',
            'hidden_layer_sizes': [[18,24,18], [20, 50, 20]]}




def export_model(model, pv_type, tuned):
    """
    Exports model using pickle
    :param model: model to be exported
    :param filename: string of filename to export model to
    :return: none
    """
    filename = str(model.__class__.__name__)
    if tuned:
        filename += "_tuned"
    directory = '../res/' + pv_type
    filepath = directory + '/'+filename +'.pkl'
    if not os.path.exists(directory):
        os.makedirs(directory)
    pickle.dump(model, open(filepath, 'wb'))


def calc_scatter_index(rmse, y_preds, model):
    # scatter index
    avg_obs = 0
    for pred in y_preds:
        # check if pred is a list in case of multi output
        if type(pred) is np.ndarray:
            for p in pred:
                avg_obs += p
        else:
            avg_obs += pred
    avg_obs = avg_obs / len(y_preds)
    si = (rmse / avg_obs) * 100

    print("SI of model " + str(model.__class__.__name__) + ": " + str(si))

    return si


def build_models(model, xtrain, ytrain, xtest, ytest):
    """
    Trains and tests given model using given test and training sets; Calculates RMSE.
    :param model: model to train
    :param xtrain: training set of x to use
    :param ytrain: training set of y to use
    :param xtest: test set of x tp use
    :param ytest: test set of y to use
    :return: trained model, rmse score, si score
    """
    # make model
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    # evaluate results
    rmse = root_mean_squared_error(ytest, y_pred)
    print("RMSE of non-tuned " + str(model.__class__.__name__) + ": " + str(rmse))

    # scatter index
    si = calc_scatter_index(rmse, y_pred, model)

    model_names.append(str(model.__class__.__name__))
    rmse_list.append(rmse)
    si_list.append(si)

    # add model with params to params dict

    return model, rmse, si


def tune_models(model, param_grid, xtrain, ytrain, xtest, ytest, cv=15, verbose=0,
                scoring='neg_root_mean_squared_error'):
    """
       Trains, tunes, and tests given model using given test and training sets; Calculates RMSE.
       :param param_grid: dictionary of parameters to test with grid search
       :param model: model to train
       :param xtrain: training set of x to use
       :param ytrain: training set of y to use
       :param xtest: test set of x tp use
       :param ytest: test set of y to use
       :param cv: integer to use for cross validation amount
       :return: tuned model, rmse score, si score
       """
    # apply hyperparameter tuning
    gridsearch = GridSearchCV(model, param_grid=param_grid, cv=cv, verbose=verbose, scoring=scoring)
    warnings.filterwarnings("ignore")
    gridsearch.fit(xtrain, ytrain)
    print("Best Parameters of " + str(model.__class__.__name__) + " are: " + str(gridsearch.best_params_))
    hypertuned_model = gridsearch.best_estimator_
    hypertuned_model.fit(xtrain, ytrain)
    y_pred_tuned = hypertuned_model.predict(xtest)

    # evaluate results
    rmse_tuned = root_mean_squared_error(ytest, y_pred_tuned)
    print("RMSE of tuned " + str(model.__class__.__name__) + ": " + str(rmse_tuned))

    # scatter index
    si_tuned = calc_scatter_index(rmse_tuned, y_pred_tuned, model)

    model_names.append(str(model.__class__.__name__) + " Tuned")
    rmse_list.append(rmse_tuned)
    si_list.append(si_tuned)

    return hypertuned_model, rmse_tuned, si_tuned


def export_to_csv(pv_type):
    results_path = '../res/' + pv_type+ '/'+ 'model_results.csv'
    results_data = {'Model Name': model_names, 'Model RMSE': rmse_list, 'Model Scatter Index': si_list}
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path, index=False)



filepath = r"Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch Simulation_0815\Canopy_BatchResults_Semi Opaque Panels.xlsx"
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11)

model_df = preprocess.process_model_data(df)
print(model_df.info)

# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)
pv_type = filepath.split('\\')[-1].replace(".xlsx", "")
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

nn_model= MLPRegressor(**nn_params)
tuned_nn_model = MLPRegressor(**nn_params_cv)

# add to parameter dictionary
model_params[str(svr_model.__class__.__name__)] = {"model": svr_model, "param_grid":svr_params}
model_params[str(tuned_svr_model.__class__.__name__) + "_tuned"] = {"model": tuned_svr_model, "param_grid":svr_params_cv}

model_params[str(ridge_model.__class__.__name__)] = {"model": ridge_model, "param_grid":ridge_params}
model_params[str(tuned_ridge_model.__class__.__name__) + "_tuned"] = {"model": tuned_ridge_model, "param_grid":ridge_params_cv}

model_params[str(lasso_model.__class__.__name__)] = {"model": lasso_model, "param_grid":lasso_params}
model_params[str(tuned_lasso_model.__class__.__name__) + "_tuned"] = {"model": tuned_lasso_model, "param_grid":lasso_params_cv}

model_params[str(nn_model.__class__.__name__)] = {"model": nn_model, "param_grid":nn_params}
model_params[str(tuned_nn_model.__class__.__name__) + "_tuned"] = {"model": tuned_nn_model, "param_grid":nn_params_cv}




#tuned_svr, tuned_svr_rmse, tuned_svr_si = tune_models(SVR(), svr_params_cv, X_train, y_train, X_test, y_test, verbose=4)

# export
#export_model(tuned_svr, '../res/tuned_svr.pkl', True)

#for model in model_params.keys():






#svr_model, svr_rmse, svr_si = build_models(svr_model, X_train, y_train, X_test, y_test)

# export
#export_model(svr_model,pv_type, False)
for model in model_params.keys():
    curr_model = model_params.get(model).get("model")
    curr_param = model_params.get(model).get("param_grid")
    skip_model = check_models_to_run(curr_model, curr_param, pv_type)
    if not skip_model:
        if 'tuned' in model:
            train_model, rmse_score, si_score = tune_models(curr_model, curr_param, X_train,y_train, X_test, y_test)
            export_model(train_model, pv_type, True)
            add_model_params(pv_type, curr_param, model, rmse_score, si_score )
        else:
            train_model, rmse_score, si_score = build_models(curr_model, X_train,y_train, X_test, y_test)
            export_model(train_model, pv_type, False)
            add_model_params(pv_type, curr_param, model, rmse_score, si_score)

export_to_csv(pv_type)


