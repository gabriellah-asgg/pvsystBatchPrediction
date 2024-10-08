from sklearn.preprocessing import StandardScaler

from src.preprocessData import Preprocessor
from src import preprocessData as preprocess
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import root_mean_squared_error
import pickle
import os
from jsonWriter import *

import warnings


class ModelBuilder:
    def __init__(self, source_data_fp, target, columns=None, sheet_name=0, features=None, pv_type=None):
        if features is None:
            features = ['Sheds Tilt', 'Sheds Azim']
        self.pv_type = pv_type
        if pv_type is None:
            self.pv_type = source_data_fp.split('\\')[-1].replace(".xlsx", "")

        self.rand = 42
        self.target = target
        self.source_data_fp = source_data_fp

        # preprocess data so it is ready to run models on
        self.df = self.construct_df(columns=columns, sheet_name=sheet_name)

        # standardize data
        self.scaler = StandardScaler()
        self.export_model(self.scaler, False)

        # split data into sets
        self.y = self.df[self.target]
        self.X = self.df[features]
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.30, random_state=self.rand)

        self.model_names = []
        self.rmse_list = []
        self.si_list = []

    def export_model(self, model, tuned, score=""):
        """
        Exports model using pickle
        :param model: model to be exported
        :param filename: string of filename to export model to
        :return: none
        """
        filename = str(model.__class__.__name__)
        if tuned:
            filename += "_tuned"
        directory = '../res/' + self.pv_type
        filepath = directory + '/' + filename + '.pkl'
        if not os.path.exists(directory):
            os.makedirs(directory)
        pickle.dump(model, open(filepath, 'wb'))

    def construct_df(self, columns=None, sheet_name=0):
        # construct dataframe for model building
        preprocessor = Preprocessor(self.source_data_fp)
        if ".xlsx" in self.source_data_fp:
            df = preprocessor.read_worksheet(skip_rows=11, columns=columns, sheet=sheet_name)

        else:
            df = preprocessor.read_csv(skip_rows=11, columns=columns)
        model_df = preprocess.process_model_data(df)
        print(model_df.info)
        return model_df

    def build_models(self, model):
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
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        # evaluate results
        rmse = root_mean_squared_error(self.y_test, y_pred)
        print("RMSE of non-tuned " + str(model.__class__.__name__) + ": " + str(rmse))

        # scatter index
        si = calc_scatter_index(rmse, y_pred, model)

        self.model_names.append(str(model.__class__.__name__))
        self.rmse_list.append(rmse)
        self.si_list.append(si)

        # add model with params to params dict

        return model, rmse, si

    def build_tf_models(self, model, compile_params, fit_params):
        """function to build tensorflow models, as the architecture is not the same as keras"""
        # make model
        model.compile(**compile_params)
        model.fit(self.X_train, self.y_train, **fit_params)
        y_pred = model.predict(self.X_test)

        # evaluate results
        rmse = root_mean_squared_error(self.y_test, y_pred)
        print("RMSE of non-tuned " + str(model.__class__.__name__) + ": " + str(rmse))

        # scatter index
        si = calc_scatter_index(rmse, y_pred, model)

        self.model_names.append(str(model.__class__.__name__))
        self.rmse_list.append(rmse)
        self.si_list.append(si)

        # add model with params to params dict

        return model, rmse, si

    def tune_models(self, model, param_grid, cv=15, verbose=0, scoring='neg_root_mean_squared_error'):
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
        gridsearch.fit(self.X_train, self.y_train)
        print("Best Parameters of " + str(model.__class__.__name__) + " are: " + str(gridsearch.best_params_))
        hypertuned_model = gridsearch.best_estimator_
        hypertuned_model.fit(self.X_train, self.y_train)
        y_pred_tuned = hypertuned_model.predict(self.X_test)

        # evaluate results
        rmse_tuned = root_mean_squared_error(self.y_test, y_pred_tuned)
        print("RMSE of tuned " + str(model.__class__.__name__) + ": " + str(rmse_tuned))

        # scatter index
        si_tuned = calc_scatter_index(rmse_tuned, y_pred_tuned, model)

        self.model_names.append(str(model.__class__.__name__) + " Tuned")
        self.rmse_list.append(rmse_tuned)
        self.si_list.append(si_tuned)

        return hypertuned_model, rmse_tuned, si_tuned, gridsearch.best_params_

    def run_model_builder(self,model_params,json_filepath):
        # run models that haven't previously been run
        for model in model_params.keys():
            curr_model = model_params.get(model).get("model")
            curr_param = model_params.get(model).get("param_grid")
            tuned = 'tuned' in model
            skip_model = check_models_to_run(curr_model, curr_param, self.pv_type, tuned, filepath=json_filepath)
            if not skip_model:
                if tuned:
                    train_model, rmse_score, si_score, best_params = self.tune_models(curr_model,curr_param,verbose=5)
                    score = "_" + str(round(rmse_score, 2))
                    export = add_model_params(pv_type, curr_param, model, rmse_score, si_score, best_params,
                                              filepath=json_filepath)

                    # only export tuned model if it is best tuned model
                    if export:
                        self.export_model(train_model,True)

                else:
                    if model_params.get(model).get("fit_params"):
                        compile_params = model_params.get(model).get("compile_params")
                        fit_params = model_params.get(model).get("fit_params")
                        train_model, rmse_score, si_score = self.build_tf_models(curr_model,compile_params,fit_params)
                        callback_params = model_params.get(model).get("fit_params").get("callbacks")[0]
                        fit_params['callbacks'] = {'monitor': callback_params.monitor,
                                                   'patience': callback_params.patience}
                        score = "_" + str(round(rmse_score, 2))
                        layer_configs = []
                        for layer in model_params.get(model).get("param_grid"):
                            layer_config = layer.get_config()
                            configs = {'units': layer_config['units'], 'activation': layer_config['activation']}
                            layer_configs.append(configs)
                        all_params = {"layers": layer_configs}
                        serial_compile_params = {'loss': compile_params.get('loss'),
                                                 'optimizer': [compile_params.get('optimizer').get_config()['name'],
                                                               compile_params.get('optimizer').get_config()[
                                                                   'learning_rate'],
                                                               compile_params.get('optimizer').get_config()['epsilon']]}
                        all_params.update(serial_compile_params)
                        all_params.update(fit_params)
                        add_model_params(self.pv_type, all_params, model, rmse_score, si_score, filepath=json_filepath)

                    else:
                        train_model, rmse_score, si_score = self.build_models(curr_model,)
                        score = "_" + str(round(rmse_score, 2))
                        add_model_params(self.pv_type, curr_param, model, rmse_score, si_score, filepath=json_filepath)

                    self.export_model(train_model,False)
        #export_to_csv(pv_type, filepath=json_filepath)


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


def serialize_parameters(params, key_params, dict):
    element_configs = []
    for element in params:
        element_config = element.get_config()
        configs = {}
        for key_p in key_params:
            configs[key_p] = element_config[key_p]
        element_configs.append(configs)
