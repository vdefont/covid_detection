import xgboost as xgb
import pandas as pd
from pandas import DataFrame
import numpy as np
import  itertools
from typing import Any, Tuple, Optional, List, Dict, NamedTuple, Callable, Iterable
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from importlib import reload

import const
import make_data_class
import train_class


# DATA #


class TrainValid(NamedTuple):
    is_binary: bool
    X_train: DataFrame
    y_train: DataFrame
    X_valid: Optional[DataFrame] = None
    y_valid: Optional[DataFrame] = None


class ClassNegPreds(NamedTuple):
    class_preds: DataFrame
    neg_preds: DataFrame


def _get_model_preds(model_name: str, is_test: bool, single: bool) -> List[ClassNegPreds]:
    """
    'single': Return just the single df corresponding to avg preds
    'all': Return list of dfs, one per fold
    """
    class_path = const.subdir_preds_class(path=True) / model_name / ("test" if is_test else "valid")
    neg_path = const.subdir_preds_neg(path=True) / model_name / ("test" if is_test else "valid")
    class_base = str(class_path)
    neg_base = str(neg_path)

    if single:
        class_preds = [pd.read_csv(f"{class_base}.csv", index_col=0)]
        neg_preds = [pd.read_csv(f"{neg_base}.csv", index_col=0)]
    else:
        with open(f"{class_base}_all", "rb") as f:
            class_preds = pickle.load(f)
        with open(f"{neg_base}_all", "rb") as f:
            neg_preds = pickle.load(f)

    # Clean the dataframes
    for i, class_preds_i in enumerate(class_preds):
        class_preds[i] = class_preds_i.rename(columns=lambda c: f"{model_name}_class_{c}")
    for i, neg_preds_i in enumerate(neg_preds):
        neg_preds_i = neg_preds_i[['negative']]  # The second columns is redundant since they sum to 1
        neg_preds[i] = neg_preds_i.rename(columns=lambda c: f"{model_name}_neg_{c}")

    return [
        ClassNegPreds(class_preds=class_preds_i, neg_preds=neg_preds_i)
        for class_preds_i, neg_preds_i in zip(class_preds, neg_preds)
    ]


def get_X(model_preds: Dict[str, ClassNegPreds], keep_study_id: bool = False, is_test: bool = False) -> DataFrame:
    # Add NN feats
    preds_parts = []
    for model_name, preds in model_preds.items():
        preds_parts.append(pd.concat([preds.class_preds, preds.neg_preds], axis=1))
    preds = pd.concat(preds_parts, axis=1)

    # Add meta feats

    feats_csv = "metadata_feats_test.csv" if is_test else "metadata_feats_train.csv"
    meta = pd.read_csv(const.subdir_data_csv() + feats_csv)
    X = preds.merge(meta, left_index=True, right_on="image_id")
    X = X.set_index("image_id")

    # Unused by xgb
    feats_to_remove = [
        'image_type_DERIVED', 'image_type_100000', 'PhotometricInterpretation_MONOCHROME2', 'part_pecho',
        'Modality_CR', 'InstanceNumber_3+', 'image_type_POST_PROCESSED',
        'SpecificCharacterSet_ISO_IR 192',
    ]
    for feat in feats_to_remove:
        assert feat in X, f"{feat} not found"
        del X[feat]

    if not keep_study_id:
        del X["study_id"]

    return X


def get_train_valid(model_names: Iterable[str], tr_vl_split: bool = True, cls: Optional[const.Vocab] = None) -> TrainValid:
    model_preds = {}
    for model_name in model_names:
        model_preds_raw = _get_model_preds(model_name=model_name, is_test=False, single=True)
        assert len(model_preds_raw) == 1
        model_preds[model_name] = model_preds_raw[0]

    X = get_X(model_preds=model_preds, keep_study_id=True)

    # Get labels

    labels = pd.read_csv(const.subdir_data_csv() + "train_study_level_prep.csv")
    labels = labels.rename(columns=dict(zip(const.VOCAB_FULL, const.VOCAB_LONG)))
    if cls is not None:
        labels["y"] = labels[const.VOCAB_SHORT_TO_LONG[cls.value]]
    else:
        labels["y"] = labels[const.VOCAB_LONG].to_numpy().argmax(1)

    labels = labels[["id", "y"]]

    y = (X.reset_index()[["image_id", "study_id"]]
         .merge(labels, left_on="study_id", right_on="id")
         .set_index("image_id")
         ['y'])
    # We were only keeping this to merge with y. Don't need it now.
    del X["study_id"]

    # Optionally split into train and valid

    if not tr_vl_split:
        return TrainValid(
            is_binary=cls is not None,
            X_train=X,
            y_train=y,
        )

    tr, vl = make_data_class.get_tr_vl(valid_amt=0.3)
    return TrainValid(
        is_binary=cls is not None,
        X_train=X.loc[tr],
        y_train=y.loc[tr],
        X_valid=X.loc[vl],
        y_valid=y.loc[vl],
    )



def get_xgb_folds_OLD(tv: TrainValid, num_folds: int) -> List[Tuple[List[int], List[int]]]:
    assert tv.X_valid is None and tv.y_valid is None, "valid data should be None when using folds"

    folds_str = make_data_class.get_folds(num_folds=num_folds)
    folds = [tv.X_train.index.get_indexer(fold) for fold in folds_str]
    ret = []
    for i, vl in enumerate(folds):
        fold_is = np.delete(np.arange(num_folds), i)
        tr = np.concatenate([folds[i] for i in fold_is])
        ret.append((tr, vl))
    return ret


def get_train_valid_folds(model_names: Iterable[str], num_folds: int, cls: Optional[const.Vocab]) -> List[TrainValid]:
    model_names = sorted(model_names)  # Aids reproducibility
    tv = get_train_valid(model_names=model_names, tr_vl_split=False, cls=cls)
    assert tv.X_valid is None and tv.y_valid is None, "shouldn't have valid data when making folds"

    folds = [list(fold) for fold in make_data_class.get_folds(num_folds=num_folds, seed=47)]
    ret = []
    for i, vl in enumerate(folds):
        fold_is = np.delete(np.arange(num_folds), i)
        tr = np.concatenate([folds[i] for i in fold_is])
        X_train = tv.X_train.loc[tr]
        y_train = tv.y_train.loc[tr]
        X_valid = tv.X_train.loc[vl]
        y_valid = tv.y_train.loc[vl]
        ret.append(TrainValid(
            is_binary=tv.is_binary, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid
        ))
    return ret


# MODEL #


def params_4class():
    lr = 0.003
    max_depth = 3
    l1_reg = 0.5
    l2_reg = 0.5
    subsample = 0.1 # Regularize (1. is nothing)
    colsample_bytree = 0.75
    colsample_bylevel = 1.

    return {
        'max_depth': max_depth, 'eta': lr, 'alpha': l1_reg, 'subsample': subsample,
        'lambda': l2_reg, "colsample_bytree": colsample_bytree, "colsample_bylevel": colsample_bylevel,
    }


def params_2class():
    lr = 0.003
    max_depth = 4
    l1_reg = 0.5
    l2_reg = 0.5
    subsample = 0.1 # Regularize (1. is nothing)
    colsample_bytree = 0.75
    colsample_bylevel = 1.

    return {
        'max_depth': max_depth, 'eta': lr, 'alpha': l1_reg, 'subsample': subsample,
        'lambda': l2_reg, "colsample_bytree": colsample_bytree, "colsample_bylevel": colsample_bylevel,
    }


def train_bst(tv: TrainValid, params_func: Callable) -> Any:
    params = params_func()

    num_rounds = 10_000
    early_stop = 30

    if tv.is_binary:
        params['objective'] = 'binary:logistic'
        # params['objective'] = 'rank:pairwise'
    else:
        params['objective'] = 'multi:softprob'
        params['num_class'] = 4

    dtrain = xgb.DMatrix(tv.X_train, tv.y_train)
    X_valid, y_valid = tv.X_valid, tv.y_valid
    assert X_valid is not None and y_valid is not None, "without folds, valid data should exist"
    dvalid = xgb.DMatrix(tv.X_valid, tv.y_valid)
    evals = [(dtrain, 'train'), (dvalid, 'valid')]

    bst = xgb.train(
        # Base
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        early_stopping_rounds=early_stop,
        # Extra
        evals=evals,
    )

    return bst


def train_bst_folds(tv_folds: List[TrainValid], params_func: Callable) -> List[Any]:
    bsts = [train_bst(tv, params_func=params_func) for tv in tv_folds]
    print(f"Per fold")
    accs = [bst_acc(tv=tv, bst=bst, use_map=True) for tv, bst in zip(tv_folds, bsts)]
    print(f"Overall: {np.mean(accs)}")
    return bsts


def _get_preds_path(model_name: str) -> Path:
    return const.subdir_preds_xgb(path=True) / model_name / "test.csv"


def save_bst_folds(bst_folds: List[Any], model_name: str) -> None:
    assert model_name.startswith("xgb"), "Must start with xgb due to dumb logic in predict_final.make_predictions"
    dir = const.subdir_models_xgb(path=True)
    if not dir.exists():
        dir.mkdir()
    with open(dir / model_name, "wb") as f:
        pickle.dump(bst_folds, f)


def predict_and_save_bst_folds(
        model_name: str,
        src_model_names: Iterable[str],
        pred_neg: bool,
        is_test: bool = True,
        all_fold_combos: bool = True,
) -> pd.DataFrame:
    acc = None
    count = 0

    model_path = const.subdir_models_xgb(path=True) / model_name
    with open(model_path, "rb") as f:
        bsts = pickle.load(f)

    preds_per_model = [
        _get_model_preds(model_name=src_model_name, is_test=is_test, single=not all_fold_combos)
        for src_model_name in src_model_names
    ]

    for preds in itertools.product(*preds_per_model):
        model_preds = dict(zip(src_model_names, preds))
        X = get_X(model_preds=model_preds, is_test=is_test)

        ret = 0.
        for bst in bsts:
            ret += bst.predict(xgb.DMatrix(X))
        ret = ret / len(bsts)

        if pred_neg:
            ret_df = pd.DataFrame({'negative': ret, 'positive': (1 - ret)})
        else:
            ret_df = pd.DataFrame(ret, columns=const.VOCAB_SHORT)

        acc = ret_df if acc is None else acc + ret_df
        count += 1

    ret = acc / count

    out_path = _get_preds_path(model_name=model_name)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True)
    ret.index = X.index
    ret.index.name = ""
    ret.to_csv(out_path)
    return ret


# METRICS #


def eval_map(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    return "mAP", train_class.preds_map(preds=predt, targs=y)


def eval_map_binary(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    y = dtrain.get_label()
    return "mAP", train_class.single_map(probs=predt, targs=y)


def bst_acc_binary(tv, bst):
    preds = bst.predict(xgb.DMatrix(tv.X_valid, tv.y_valid))
    targs = tv.y_valid
    return train_class.single_map(probs=preds, targs=targs)


def bst_acc(tv, bst, use_map: bool = True):
    preds = bst.predict(xgb.DMatrix(tv.X_valid))
    targs = tv.y_valid
    if use_map:
        if tv.is_binary:
            acc = train_class.single_map(probs=preds, targs=targs)
        else:
            acc = train_class.preds_map(preds=preds, targs=targs)
    else:
        if tv.is_binary:
            acc = -np.mean([np.log(pred if targ == 1 else 1 - pred) for pred, targ in zip(preds, targs)])
        else:
            acc = -np.log(preds[range(len(targs)), targs]).mean()
    print(acc)
    return acc


def eval_baseline(tv):
    # Seeing what the mAP for *just* the inputs are (no training)
    preds = tv.X_valid[[f'class_{i}' for i in range(4)]].to_numpy()
    targs = tv.y_valid
    print(train_class.preds_map(preds=preds, targs=targs))


# EXPERIMENTS #


"""
Experimental results indicate that:
- one four-class mAP with 'multi:softprob' works just as well as four binary with binary:logitraw
- For binary, using rank:pairwise is about the same. Using rank:map doesn't seem to train!
"""


def eval_one(tv):
    bst = train_bst(tv)
    bst_acc(tv, bst, use_map=True)


def eval_multi(tv_neg, tv_typ, tv_ind, tv_atyp):
    tvs = {'neg': tv_neg, 'typ': tv_typ, 'ind': tv_ind, 'atyp': tv_atyp}

    bsts = {}
    for lab, tv in tvs.items():
        print(f"Training {lab}")
        bsts[lab] = train_bst(tv=tv)

    acc_cum = 0.
    for lab, tv in tvs.items():
        bst = bsts[lab]
        acc_i = bst_acc_binary(tv=tv, bst=bst)
        acc_cum += acc_i
        print(f"{lab} mAP: {acc_i}")
    map = acc_cum / 4
    print(f"Overall mAP: {map} (2/3: {map * 2 / 3})")


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


# OTHER SKLEARN ALGORITHMS #


def accuracy(tv, cl):
    y = tv.y_valid.to_numpy()
    y_hat = cl.predict_proba(tv.X_valid)
    print(train_class.preds_map(preds=y_hat, targs=y))


def train_xtr(tv):
    xtr = ExtraTreesClassifier(n_estimators=500, n_jobs=-1)
    xtr.fit(tv.X_train, tv.y_train)
    accuracy(tv, xtr) # 0.813 / 0.611
    return xtr


def train_svm(tv):
    svm_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", LinearSVC(max_iter=10_000)),
    ])
    svm_lin.fit(tv.X_train, tv.y_train)
    accuracy(tv, svm_lin) # 0.829 / 0.623
    return svm_lin


def train_svm_poly(tv):
    svm_poly = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="poly", degree=3, coef0=1, C=5)),
    ])
    svm_poly.fit(tv.X_train, tv.y_train)
    accuracy(tv, svm_poly) # 0.821 / 0.616
    return svm_poly


def train_svm_gauss(tv):
    svm_gauss = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", gamma=5, C=0.001)),
    ])
    svm_gauss.fit(tv.X_train, tv.y_train)
    accuracy(tv, svm_gauss) # 0.726 / 0.475 (BAD!)
    return svm_gauss