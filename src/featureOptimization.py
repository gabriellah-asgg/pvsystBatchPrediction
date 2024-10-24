from bayes_opt import BayesianOptimization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pickle as pkl


def objective_function(tilt, azim):
    """Function to learn the objective function of the model; returns negative mean squared error"""
    best_model = pkl.load(open(r'../res/earray_data/SVR_tuned.pkl', 'rb'))
    inputs = np.array([tilt, azim])
    predictions = best_model.predict(inputs)
    return predictions




