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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings

# standard random state
rand = 42

# export model
def export_model(model, filename):
    """
    Exports model using pickle
    :param model: model to be exported
    :param filename: string of filename to export model to
    :return: none
    """
    pickle.dump(model, open(filename, 'wb'))


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

    return model, rmse, si


def tune_models(model, param_grid, xtrain, ytrain, xtest, ytest, cv=15, verbose=0, scoring='neg_root_mean_squared_error'):
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

def export_to_csv():
    results_path = '../res/model_results.csv'
    results_data = {'Model Name': model_names, 'Model RMSE': rmse_list, 'Model Scatter Index': si_list}
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(results_path, index=False)


# construct dataframe for model building
filepath = (r'Q:\Projects\224008\DESIGN\ANALYSIS\00_PV\PVsyst Batch Simulation\PV Batch '
            r'Simulation_0815\Canopy_Section_A_BatchResults_all_Panels.xlsx')
preprocessor = Preprocessor(filepath)
df = preprocessor.read_worksheet(skip_rows=11)

model_df = preprocess.process_model_data(df)
print(model_df.info)

# standardize data
scaler = StandardScaler()
y = model_df['EArray (KWh)']
X = model_df.drop(['EArray (KWh)'], axis=1)
X_scaled = scaler.fit_transform(X)
export_model(scaler, '../res/scaler.pkl')

# split for train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=rand)

# create data structures to capture model results
model_names = []
rmse_list = []
si_list = []


# build models
# make svm model
svr_model, svr_rmse, svr_si = build_models(SVR(), X_train, y_train, X_test, y_test)

# export
export_model(svr_model, '../res/svr.pkl')

# linear regression model
ridge_model, ridge_rmse, ridge_si = build_models(Ridge(), X_train, y_train, X_test, y_test)

# export
export_model(ridge_model, '../res/ridge.pkl')


# Lasso model
lasso_model, lasso_rmse, lasso_si = build_models(linear_model.Lasso(), X_train, y_train, X_test, y_test)

# export
export_model(lasso_model, '../res/lasso.pkl')

# neural network
nn = MLPRegressor(random_state=rand, activation='relu', alpha=10, hidden_layer_sizes=[18, 24, 18],
                  learning_rate='constant', learning_rate_init=0.1, solver='adam')
nn, nn_rmse, nn_si = build_models(nn, X_train, y_train, X_test, y_test)

# export
export_model(nn, '../res/nn.pkl')


export_to_csv()

# hyper parameter tuning
svr_param_grid = {'kernel': ('linear', 'rbf', 'sigmoid', 'poly'),
                  'C': [1, 10]}
tuned_svr, tuned_svr_rmse, tuned_svr_si = tune_models(SVR(), svr_param_grid, X_train, y_train, X_test, y_test, verbose=4)

# export
export_model(tuned_svr, '../res/tuned_svr.pkl')

# hyper tuned ridge regression
param_ridge = {'solver': ('auto', 'svd', 'lsqr'),
               'alpha': [1, 1.5, 5, 10, 20, 50]}
tuned_ridge, tuned_ridge_rmse, tuned_rdige_si = tune_models(Ridge(), param_ridge, X_train, y_train, X_test, y_test, verbose=4)

# export
export_model(tuned_ridge, '../res/tuned_ridge.pkl')

# hyper tuned lasso regression
param_lasso = {'alpha': [0.01, 0.1, 0.5, 1, 1.5, 5, 10, 20, 50]}
tuned_lasso_model, tuned_lasso_rmse, tuned_lasso_si = tune_models(linear_model.Lasso(), param_lasso, X_train, y_train, X_test, y_test, verbose=4)

# export
export_model(tuned_lasso_model, '../res/tuned_lasso.pkl')

export_to_csv()