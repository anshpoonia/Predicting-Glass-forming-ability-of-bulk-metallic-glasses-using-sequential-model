import os
import sklearn
import seaborn as sns
import numpy as np
import pandas as pd
import re
import pickle
import json
import math
import random
from tqdm import tqdm
import datetime
import time

from PyAstronomy import pyasl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    # RobustScalar,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoder, 
    OrdinalEncoder,
    LabelEncoder
)

from sklearn.utils import all_estimators

from sklearn.base import (
    RegressorMixin, 
    ClassifierMixin,
    TransformerMixin
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    auc,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
    classification_report
)

import warnings
import xgboost
import catboost
import lightgbm

import tensorflow as tf

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    # "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    # "MLPClassifier",
    "LogisticRegressionCV", 
    "MultiOutputClassifier", 
    "MultinomialNB", 
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
    "CategoricalNB",
    "StackingClassifier",
    "NuSVC",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression", 
    "CCA", 
    "IsotonicRegression", 
    "StackingRegressor",
    "MultiOutputRegressor", 
    "MultiTaskElasticNet", 
    "MultiTaskElasticNetCV", 
    "MultiTaskLasso", 
    "MultiTaskLassoCV", 
    "PLSCanonical", 
    "PLSRegression", 
    "RadiusNeighborsRegressor", 
    "RegressorChain", 
    "VotingRegressor", 
    "QuantileRegressor"
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]


REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))
REGRESSORS.append(('CatBoostRegressor', catboost.CatBoostRegressor))

CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))
CLASSIFIERS.append(('CatBoostClassifier', catboost.CatBoostClassifier))

TRANSFOMER_METHODS = [
    ("StandardScaler", StandardScaler), 
    ("MinMaxScaler", MinMaxScaler), 
    ("MaxAbsScaler", MaxAbsScaler), 
    # ("RobustScalar", RobustScalar),
    ("Normalizer", Normalizer),
    ("QuantileTransformer", QuantileTransformer),
    ("PowerTransformer", PowerTransformer),
]

def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))


def run_all_regressors(X_train, y_train, X_test, y_test):
    R2 = []
    ADJR2 = []
    RMSE = []
    names = []
    TIME = []
    for name, model in tqdm(REGRESSORS):
        start = time.time()
        pipe = Pipeline(steps=[
                            ("classifier", model()),
                        ]
                    )
        try:
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            print(name, r_squared)
            adj_rsquared = adjusted_rsquared(
                r_squared, X_test.shape[0], X_test.shape[1]
            )
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            names.append(name)
            R2.append(r_squared)
            ADJR2.append(adj_rsquared)
            RMSE.append(rmse)
            TIME.append(time.time() - start)
        except Exception as exception:
            print(name + " model failed to execute")
            print(exception)
        scores = {
                    "Model": names,
                    "Adjusted R-Squared": ADJR2,
                    "R-Squared": R2,
                    "RMSE": RMSE,
                    "Time Taken": TIME,
                }
        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by = "Adjusted R-Squared", ascending = False).set_index("Model")
    display(scores[:])

def run_all_regressors_with_transformers(X_train, y_train, X_test, y_test):
    R2 = []
    ADJR2 = []
    RMSE = []
    names = []
    TIME = []
    Model_params = []

    for transformer_method_name, transformer_method in tqdm(TRANSFOMER_METHODS):
        for name, model in tqdm(REGRESSORS):
            start = time.time()
            X_transformer = transformer_method()
            y_transformer = transformer_method()
            transformed_X_train = pd.DataFrame(X_transformer.fit_transform(X_train), columns = X_train.columns)
            transformed_X_test = pd.DataFrame(X_transformer.transform(X_test), columns = X_test.columns)

            transformed_y_train = pd.DataFrame(y_train, columns = y_train.columns)
            transformed_y_test = pd.DataFrame(y_test, columns = y_test.columns)
            pipe = Pipeline(steps=[
                                ("classifier", model()),
                            ]
                        )
            try:
                pipe.fit(transformed_X_train, transformed_y_train)
                transformed_y_pred = pipe.predict(transformed_X_test)
                r_squared = r2_score(transformed_y_test, transformed_y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, transformed_X_test.shape[0], transformed_X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(transformed_y_test, transformed_y_pred))
                names.append(name + " (" + transformer_method_name + ")")
                print(name + " (" + transformer_method_name + ")", r_squared)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)
                Model_params.append(str(pipe.get_params()))
            except Exception as exception:
                print(name + " (" + transformer_method_name + ")" + " model failed to execute")
                print(exception)
    scores = {
                "Model": names,
                "Adjusted R-Squared": ADJR2,
                "R-Squared": R2,
                "RMSE": RMSE,
                "Time Taken": TIME,
                "Model Params": Model_params,
            }
    scores = pd.DataFrame(scores)
    scores = scores.sort_values(by = "Adjusted R-Squared", ascending = False).set_index("Model")
    return scores