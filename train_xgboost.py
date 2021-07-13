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
import matplotlib.pyplot as plt

import const
import make_data_class


class TrainValid(NamedTuple):
    X_train: DataFrame
    y_train: DataFrame
    X_valid: DataFrame
    y_valid: DataFrame


def get_train_valid(is_neg: bool) -> TrainValid:
    # Add NN feats

    preds_class = pd.read_csv(const.subdir_preds_class(path=True)/'resnet18'/'valid.csv', index_col=0)
    preds_class = preds_class.rename(columns=lambda c: f"class_{c}")

    preds_neg = pd.read_csv(const.subdir_preds_neg(path=True)/'resnet18'/'valid.csv', index_col=0)
    preds_neg = preds_neg[['0']]  # The second columns is redundant since they sum to 1
    preds_neg = preds_neg.rename(columns=lambda c: f"neg_{c}")

    preds = pd.concat([preds_class, preds_neg], axis=1)

    # Add meta feats

    meta = pd.read_csv(const.subdir_data_csv() + "metadata_feats_train.csv")
    X = preds.merge(meta, left_index=True, right_on="image_id")
    X = X.set_index("image_id")

    # Unused by xgb
    feats_to_remove = [
        'image_type_DERIVED', 'image_type_100000', 'PhotometricInterpretation_MONOCHROME2', 'part_pecho',
        'Modality_CR', 'InstanceNumber_3+', 'PatientSex_F', 'image_type_POST_PROCESSED',
        'SpecificCharacterSet_ISO_IR 192',
    ]
    for feat in feats_to_remove:
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

    y = (X.reset_index()[["image_id", "study_id"]]
         .merge(labels, left_on="study_id", right_on="id")
         .set_index("image_id")
         ['y'])
    # We were only keeping this to merge with y. Don't need it now.
    del X["study_id"]

    # Split into train and valid
    tr, vl = make_data_class.get_tr_vl(valid_amt=0.3)
    return TrainValid(
        X_train=X.loc[tr],
        y_train=y.loc[tr],
        X_valid=X.loc[vl],
        y_valid=y.loc[vl],
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

    return bst


def plot_importance(bst):
    xgb.plot_importance(bst)
    plt.show()


def print_importance(bst):
    imp = dict(sorted(bst.get_score().items(), key=lambda t: -t[1]))
    print(imp)


def print_unused(tv, bst):
    all_cols = set(tv.X_train.columns)
    used_cols = set(bst.get_score())
    unused_cols = all_cols - used_cols
    print(unused_cols)


# tv = get_train_valid(is_neg=False)


def accuracy(cl):
    y = tv.y_valid.to_numpy()
    y_hat = cl.predict_proba(tv.X_valid)
    pred_probs = y_hat[range(len(y)), y]
    loss = -np.log(pred_probs).mean()
    print(loss)


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