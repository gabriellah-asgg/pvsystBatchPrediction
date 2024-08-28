import pandas as pd

from preprocessData import Preprocessor
import preprocessData as preprocess
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error
import pickle
import os
import warnings


def construct_df(filepath, columns=None):
    # construct dataframe for model building
    preprocessor = Preprocessor(filepath)
    if ".xlsx" in filepath:
        df = preprocessor.read_worksheet(skip_rows=11, columns=columns)

    else:
        df = preprocessor.read_csv(skip_rows=11, columns=columns)
    model_df = preprocess.process_model_data(df)
    print(model_df.info)
    return model_df


def export_model(model, pv_type, tuned, score=""):
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
    filepath = directory + '/' + filename + '.pkl'
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


def build_models(model, xtrain, ytrain, xtest, ytest, model_names, rmse_list, si_list):
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

    return model, rmse, si, model_names, rmse_list, si_list


def tune_models(model, param_grid, xtrain, ytrain, xtest, ytest, model_names, rmse_list, si_list, cv=15, verbose=0,
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

    return hypertuned_model, rmse_tuned, si_tuned, model_names, rmse_list, si_list, gridsearch.best_params_


