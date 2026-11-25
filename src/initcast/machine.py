"""
All the methods to train a Gradient-Boosted Trees (XGBoost) model.


Methods:
    train_model(training: Training, **kwargs):
        train an XGBoost model with a 'Training' object

    objective(trial, kf, days):
        method used to run param_tune()

    param_tune(training: Training, n_splits=5, n_trials=100):
        hyperparameter tune the model with a small dataset
"""

import xarray as xr
import numpy as np
from training import Training

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
import shap

from datetime import datetime, timedelta
import random

def train_model(training: Training, **kwargs):
    n_estimators = kwargs.get('n_estimators', 282)
    max_depth = kwargs.get('max_depth', 6)
    learning_rate = kwargs.get('learning_rate', 0.014265901963262596)
    subsample = kwargs.get('subsample', 0.6532045263433063)
    colsample_bytee = kwargs.get('colsample_bytree', 0.9441442315437493)
    min_child_weight = kwargs.get('min_child_weight', 15)
    gamma = kwargs.get('gamma', 0.6073766252425523)
    reg_alpha = kwargs.get('reg_alpha', 0.9198374824724933)
    reg_lambda = kwargs.get('reg_lambda', 1.761972528776942e-06)
    scale_pos_weight = kwargs.get('scale_pos_weight', 18.279842812817886)

    model = XGBClassifier(n_estimators=n_estimators,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          subsample=subsample,
                          colsample_bytree=colsample_bytee,
                          min_child_weight=min_child_weight,
                          gamma=gamma,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          scale_pos_weight=scale_pos_weight)

    model.fit(training.data, training.target)

    return model

def objective(trial, kf, days):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 20
        ),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10, log=True
        ),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", 15, 200
        ),

        "objective": "binary:logistic",
        "eval_metric": ["aucpr", "logloss"],
    }

    scores = []
    for fold, (train_fold, test_fold) in enumerate(kf.split(days)):
        model = XGBClassifier(**params)

        train_days = [days[i] for i in train_fold]
        test_days = [days[i] for i in test_fold]

        train_fold = Training('rap', [29, 41, -105, -93], train_days, log=False)
        test_fold = Training('rap', [29, 41, -105, -93], test_days, log=False)

        model.fit(
            train_fold.data, train_fold.target, eval_set=[(test_fold.data, test_fold.target)],
            verbose=50
        )

        predictions = model.predict_proba(test_fold.data)[:, 1]

        score = average_precision_score(
            test_fold.target,
            predictions
        )

        scores.append(score)

    return np.mean(scores)

def param_tune(training: Training, n_splits=5, n_trials=100):

    if len(training.days) % n_splits != 0:
        raise ValueError('Amount of training days must be divisible by n_splits')

    days = training.days
    random.shuffle(days)

    kf = KFold(n_splits=n_splits, shuffle=False)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, kf, days),
                   n_trials=n_trials)

    print(f'Best AUC-PR Score: {round(study.best_value, 5)}')
    print(f'Best parameters:\n{study.best_params}')

    return study.best_value, study.best_params

def best_features(model: XGBClassifier, test: Training, type='beeswarm', shap_values=None):

    if shap_values is None:
        explainer = shap.TreeExplainer(model, test.data)
        shap_values = explainer(test.data)
        shap_values.feature_names = test.feature_names

    if type == 'beeswarm':
        shap.plots.beeswarm(shap_values)
    if type == 'violin':
        shap.plots.violin(shap_values)

    return shap_values

def predict(test: Training, model: XGBClassifier, hours, list=False):

    prob_list = []
    pred_list = []
    for hour in hours:
        X, _ = test.extract_hour(hour)
        prob_list.append(model.predict_proba(X)[:, 1])
        pred_list.append(model.predict(X))

    if list:
        return prob_list, pred_list

    else:
        probs = np.vstack(prob_list)
        preds = np.vstack(pred_list)

        return probs, preds

def composite_pred(test: Training, model: XGBClassifier, start_hour, hours: int):
    start_hour_datetime = datetime.strptime(start_hour, '%Y-%m-%d %H:%M')

    prob_list = []
    pred_list = []
    for i in range(hours):
        hour = start_hour_datetime + timedelta(hours=i)
        hour = hour.strftime('%Y-%m-%d %H:%M')

        print(hour)

        X, _ = test.extract_hour(hour)

        prob_list.append(1 - model.predict_proba(X)[:, 1])
        pred_list.append(model.predict(X))

    probs = 1 - np.prod(prob_list, axis=0)
    preds = np.maximum.reduce(pred_list)

    return probs, preds
