import xgboost as xgb
import pandas as pd
from pandas import DataFrame
import numpy as np
import  itertools
from typing import Tuple, Optional, List, Dict, NamedTuple
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import const


class TrainValid(NamedTuple):
    X_train: DataFrame
    y_train: DataFrame
    X_valid: DataFrame
    y_valid: DataFrame


def tv(is_neg: bool) -> TrainValid:
    # Add NN feats

    preds_class_train = pd.read_csv(const.subdir_preds_class(path=True)/'resnet18'/'train.csv', index_col=0)
    preds_class_valid = pd.read_csv(const.subdir_preds_class(path=True)/'resnet18'/'valid.csv', index_col=0)
    preds_neg_train = pd.read_csv(const.subdir_preds_neg(path=True)/'resnet18'/'train.csv', index_col=0)
    preds_neg_valid = pd.read_csv(const.subdir_preds_neg(path=True)/'resnet18'/'valid.csv', index_col=0)

    preds_train = pd.concat([preds_class_train, preds_neg_train], axis=1)
    preds_valid = pd.concat([preds_class_valid, preds_neg_valid], axis=1)

    # Add meta feats

    meta = pd.read_csv(const.subdir_data_csv() + "metadata_feats_train.csv")

    X_train = preds_train.merge(meta, left_index=True, right_on="image_id")
    X_valid = preds_valid.merge(meta, left_index=True, right_on="image_id")
    del X_train["image_id"]
    del X_valid["image_id"]
    X_train = X_train.set_index("study_id")
    X_valid = X_valid.set_index("study_id")

    feats_to_remove = []
    for feat, X in itertools.product(feats_to_remove, [X_train, X_valid]):
        assert feat in X, f"{feat} not found"
        del X[feat]

    # Get labels

    labels = pd.read_csv(const.subdir_data_csv() + "train_study_level_prep.csv")
    labels = labels.rename(columns=dict(zip(const.VOCAB_FULL, const.VOCAB_LONG)))
    if is_neg:
        labels["y"] = labels["negative"]
    else:
        labels["y"] = labels[const.VOCAB_LONG].to_numpy().argmax(1)

    labels = labels[["id", "y"]]

    y_train = X_train[[]].merge(labels, left_index=True, right_on="id")["y"]
    y_valid = X_valid[[]].merge(labels, left_index=True, right_on="id")["y"]

    return TrainValid(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
    )


def train_xgb(tv: TrainValid) -> None:
    dtrain = xgb.DMatrix(tv.X_train, tv.y_train)
    dvalid = xgb.DMatrix(tv.X_valid, tv.y_valid)

    num_rounds = 1000
    early_stop = 10
    lr = 0.03
    max_depth = 4
    l1_reg = 0.5
    l2_reg = 0.5
    subsample = 0.1 # Regularize (1. is nothing)
    colsample_bytree = 1.
    colsample_bylevel = 1.

    params = {'objective': 'multi:softprob', 'num_class': 4, 'max_depth': max_depth, 'eta': lr, 'alpha': l1_reg,
              'subsample': subsample, 'lambda': l2_reg, "colsample_bytree": colsample_bytree,
              "colsample_bylevel": colsample_bylevel}
    evals = [(dtrain, 'train'), (dvalid, 'valid')]
    bst = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=evals, early_stopping_rounds=early_stop)

    acc = (bst.predict(dvalid).argmax(1) == tv.y_valid).mean()
    print(f"Accuracy: {acc}") # 0.829
    

tv = tv(is_neg=False)


def accuracy(cl):
    y = tv.y_valid.to_numpy()
    y_hat = cl.predict(tv.X_valid)
    print((y == y_hat).mean())


def train_xtr():
    xtr = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
    xtr.fit(tv.X_train, tv.y_train)
    accuracy(xtr) # 0.813 / 0.611


def train_svm():
    svm_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(max_iter=10_000)),
    ])
    svm_lin.fit(tv.X_train, tv.y_train)
    accuracy(svm_lin) # 0.829 / 0.623


def train_svm_poly():
    svm_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", degree=3, coef0=1, C=5)),
    ])
    svm_poly.fit(tv.X_train, tv.y_train)
    accuracy(svm_poly) # 0.821 / 0.616


def train_svm_gauss():
    svm_gauss = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", gamma=5, C=0.001)),
    ])
    svm_gauss.fit(tv.X_train, tv.y_train)
    accuracy(svm_gauss) # 0.726 / 0.475 (BAD!)