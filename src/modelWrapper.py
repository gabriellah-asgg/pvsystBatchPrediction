import warnings
from abc import ABC, abstractmethod, abstractproperty

import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV


class ModelWrapper(ABC):
    """
    Wrapper class to make different model types
    """

    def __init__(self, model):
        self.model = model
        self.model_type = str(self.model.__class__.__name__)
        self.rmse = -1
        self.si = -1
        self.params = None

    @abstractmethod
    def train_model(self, params, x_train, y_train, x_test, y_test):
        pass

    @abstractmethod
    def equals(self, params):
        equal = True
        if len(self.params) != len(params):
            equal = False
        for key in params:
            # the input params may have tuples that need to be converted to lists
            if isinstance(params[key], tuple):
                params[key] = list(params[key])
            if isinstance(params[key], list):
                params[key] = sorted(params[key], key=lambda x: (x is None, x))

        for key in self.params:
            if isinstance(params[key], list):
                self.params[key] = sorted(params[key], key=lambda x: (x is None, x))
        if params == self.params:
            equal = True
        return equal

    @abstractmethod
    def serialize_parameters(self):
        pass

    def calc_scatter_index(self, rmse, y_preds, model):
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

        print("SI of model " + self.model_type + ": " + str(si))

        return si


class BaseModel(ModelWrapper, ABC):
    @abstractmethod
    def train_model(self, params, x_train, y_train, x_test, y_test):
        """
                Trains and tests given model using given test and training sets; Calculates RMSE.
                :param model: model to train
                :param x_train: training set of x to use
                :param y_train: training set of y to use
                :param x_test: test set of x to use
                :param y_test: test set of y to use
                :return: trained model, rmse score, si score
        """
        self.params = params
        # make model
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)

        # evaluate results
        self.rmse = root_mean_squared_error(y_test, y_pred)
        print("RMSE of non-tuned " + self.model_type + ": " + str(self.rmse))

        # scatter index
        self.si = self.calc_scatter_index(self.rmse, y_pred, self.model)

        return self.model_type, self.rmse, self.si


class TunedModel(ModelWrapper, ABC):
    @abstractmethod
    def train_model(self, params, x_train, y_train, x_test, y_test, cv=15, verbose=0,
                    scoring='neg_root_mean_squared_error'):
        # apply hyperparameter tuning
        gridsearch = GridSearchCV(self.model, param_grid=params, cv=cv, verbose=verbose, scoring=scoring)
        warnings.filterwarnings("ignore")
        gridsearch.fit(x_train, y_train)
        self.params = gridsearch.best_params_
        print("Best Parameters of " + self.model_type + " are: " + str(gridsearch.best_params_))
        hypertuned_model = gridsearch.best_estimator_
        hypertuned_model.fit(x_train, y_train)
        self.model = hypertuned_model
        y_pred_tuned = hypertuned_model.predict(x_test)

        # evaluate results
        self.rmse = root_mean_squared_error(y_test, y_pred_tuned)
        print("RMSE of tuned " + self.model_type + ": " + str(self.rmse))

        # scatter index
        si_tuned = self.calc_scatter_index(self.rmse, y_pred_tuned, self.model)

        return hypertuned_model, self.rmse, si_tuned, gridsearch.best_params_


class TFModel(ModelWrapper, ABC):

    @abstractmethod
    def train_model(self, params, x_train, y_train, x_test, y_test):
        self.params = params
        compile_params = params.get('compile_params')
        fit_params = params.get('fit_params')

        self.model.compile(**compile_params)
        self.model.fit(x_train, y_train, **fit_params)
        y_pred = self.model.predict(x_test)

        # evaluate results
        self.rmse = root_mean_squared_error(y_test, y_pred)
        print("RMSE of non-tuned " + self.model_type + ": " + str(self.rmse))

        # scatter index
        si = self.calc_scatter_index(self.rmse, y_pred, self.model)

        # add model with params to params dict

        return self.model, self.rmse, si

    @abstractmethod
    def serialize_parameters(self):
        compile_params = self.params.get("param_grid").get("compile_params")
        fit_params = self.params.get("param_grid").get("fit_params")
        callback_params = self.params.get("param_grid").get("fit_params").get("callbacks")[0]
        fit_params['callbacks'] = {'monitor': callback_params.monitor,
                                   'patience': callback_params.patience}
        serialized_params = {"layers": self.params.get("param_grid").get("layers")}
        serial_compile_params = {'loss': compile_params.get('loss'),
                                 'optimizer': [compile_params.get('optimizer').get_config()['name'],
                                               compile_params.get('optimizer').get_config()[
                                                   'learning_rate'],
                                               compile_params.get('optimizer').get_config()['epsilon']]}
        serialized_params.update(serial_compile_params)
        serialized_params.update(fit_params)
        return serialized_params

    @abstractmethod
    def equals(self, params):
        equal = True
        if len(self.params) != len(params):
            equal = False
            return equal
        for key in self.params.keys():
            if not params.get(key):
                equal = False
                return equal
            if len(self.params[key]) != len(params[key]):
                equal = False
                return equal
            if type(params[key]) is list:
                if sorted(self.params[key]) != sorted(self.params[key]):
                    equal = False
                    return equal
            elif self.params[key] != params[key]:
                equal = False
                return equal
        return equal