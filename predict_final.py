import pandas as pd
from pandas import DataFrame
import numpy as np
import pickle
from icevision.all import *
from typing import List, Dict, Callable, Optional, Tuple
import torch
from copy import deepcopy
from functools import partial
from pathlib import Path
import ensemble_boxes
import tqdm

import const
import preprocess
import make_data_detect
from make_data_detect import Frame, Box
import train_detect


# WEIGHTED BOX FUSION #


def _max_box_coord(boxes: List[Box]):
    return np.array([[box.X + box.W, box.Y + box.H] for box in boxes]).max()


def _prep_box(box: Box, max_coord: float) -> List[float]:
    return [c / max_coord for c in [box.X, box.Y, box.X + box.W, box.Y + box.H]]


def _prep_boxes(boxes: List[Box], max_coord: float) -> List[List[float]]:
    return [_prep_box(box, max_coord) for box in boxes]


def _get_scores(boxes: List[Box]) -> List[float]:
    return [box.C for box in boxes]


def _get_labels(boxes: List[Box]) -> List[int]:
    return [0 for box in boxes]


def _unprep_box(box: List[float], score: float, max_coord: float) -> Box:
    return Box(
        X=box[0] * max_coord,
        Y=box[1] * max_coord,
        W=(box[2] - box[0]) * max_coord,
        H=(box[3] - box[1]) * max_coord,
        C=score,
    )


def _unprep_boxes(boxes: List[List[float]], scores: List[float], max_coord: float) -> List[Box]:
    return [
        _unprep_box(box=box, score=score, max_coord=max_coord)
        for box, score in zip(boxes, scores)
    ]


def weighted_box_fusion(
        preds: List[Dict[str, List[Box]]],
        iou_thr_nms: Optional[float] = 0.4,
        iou_thr_wbf: float = 0.4,
        conf_type: str = 'max',
        skip_box_thr: float = 0.0001,
) -> Dict[str, Box]:
    ids = list(preds[0].keys())
    for preds_i in preds:
        assert preds_i.keys() == set(ids)

    max_coord = _max_box_coord(
        [box for preds_i in preds for boxes in preds_i.values() for box in boxes]
    )

    # Everything has equal weight for now
    weights = [1] * len(preds)

    ret = {}

    for id in tqdm.tqdm(ids, desc="Weighted box fusion"):
        boxes_list = [_prep_boxes(preds_i[id], max_coord=max_coord) for preds_i in preds]
        scores_list = [_get_scores(preds_i[id]) for preds_i in preds]
        labels_list = [_get_labels(preds_i[id]) for preds_i in preds]

        if iou_thr_nms is not None:
            for i, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
                boxes_list[i], scores_list[i], labels_list[i] = ensemble_boxes.nms(
                    [boxes], [scores], [labels], weights=[1], iou_thr=iou_thr_nms
                )

        # Do weighted box fusion
        boxes, scores, _labels = ensemble_boxes.weighted_boxes_fusion(
            boxes_list, scores_list, labels_list, weights=weights,
            iou_thr=iou_thr_wbf, skip_box_thr=skip_box_thr, conf_type=conf_type,
        )

        ret[id] = _unprep_boxes(boxes=boxes, scores=scores, max_coord=max_coord)

    return ret


# COMPARING DIFFERENT METHODS FOR COMBINING PREDICTIONS #


def get_fold_preds() -> List[Dict[str, List[Box]]]:
    return [
        train_detect.get_box_preds(
            preds_path_detect=const.subdir_preds_detect(path=True) / 'eff_lite0_256',
            new_frame=Frame(256, 256),
            sname=f'fold{i}',
            nms=None,
        )
        for i in range(5)
    ]


def get_valid_boxes() -> Dict[str, List[Box]]:
    img_path = const.subdir_data_detect(path=True) / "png224_7fold_test01" / "test" / "images"
    ids = [id.name.replace('.png', '') for id in img_path.glob('*png')]

    data = pd.read_csv(const.subdir_data_csv(path=True) / "train_image_level_prep.csv")
    id_to_boxes_raw = dict(zip(data.id, data.boxes))

    ret = {}
    for id in ids:
        boxes_raw = id_to_boxes_raw[id]
        boxes_json = json.loads(boxes_raw)
        ret[id] = [Box(X=e['x'], Y=e['y'], W=e['width'], H=e['height']) for e in boxes_json]
    return ret


def get_map(preds: Dict[str, List[Box]], actuals: Dict[str, List[Box]]) -> float:
    assert preds.keys() == actuals.keys()
    actuals_preds_ls = [(actuals[id], preds[id]) for id in preds]
    return train_detect.mean_average_precision(actuals_preds_ls=actuals_preds_ls)


def grid_search() -> pd.DataFrame:
    """
    Pickle saved: combine_folds_grid_search_df (combining 5 folds of eff_lite0_256)
    """
    preds = get_fold_preds()
    actuals = get_valid_boxes()

    keys = ['skip_box_thr', 'conf_type', 'iou_thr_nms', 'iou_thr_wbf']
    items = list(itertools.product(
        [0.0001, 0.001, 0.00001, 0.01],
        ['avg', 'max'],
        np.arange(0.35, 0.751, 0.05),
        np.arange(0.35, 0.751, 0.05),
    ))
    df = pd.DataFrame(dict(zip(keys, zip(*items))))
    df['mAP'] = -1

    for i, row in df.iterrows():
        fused = weighted_box_fusion(
            preds=preds,
            iou_thr_nms=row.iou_thr_nms,
            iou_thr_wbf=row.iou_thr_wbf,
            conf_type=row.conf_type,
            skip_box_thr=row.skip_box_thr,
        )
        mAP = get_map(preds=fused, actuals=actuals)
        print(f"nms {row.iou_thr_nms}, wbf {row.iou_thr_wbf}, conf {row.conf_type}, thr {row.skip_box_thr} -- {mAP}")
        df.loc[i, 'mAP'] = mAP

    return df


def inspect_best(df):
    """
    For combining folds, the best is:

    skip_box_thr conf_type  iou_thr_nms  iou_thr_wbf       mAP
    0.001        avg        0.50         0.60              0.531762
    """
    return df.sort_values(by='mAP', ascending=False).iloc[:30]


def inspect(df, skip_box_thr: float = 0.00100, conf_type: str = 'avg'):
    d = df[(df.skip_box_thr == skip_box_thr) & (df.conf_type == conf_type)]
    nms, wbf = np.meshgrid(np.arange(0.35, 0.751, 0.05), np.arange(0.35, 0.751, 0.05))
    mAP = np.zeros_like(nms)
    for i, j in itertools.product(range(nms.shape[0]), range(nms.shape[1])):
        mAP[i, j] = d[(d.iou_thr_nms == nms[i, j]) & (d.iou_thr_wbf == wbf[i, j])].iloc[0].mAP

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(nms, wbf, mAP)
    ax.set_xlabel("nms_iou")
    ax.set_ylabel("nms_wbf")
    plt.show()


def _combine_folds(fold_preds: List[Dict[str, List[Box]]]) -> Dict[str, Box]:
    return weighted_box_fusion(
        preds=fold_preds,
        iou_thr_nms=0.50,
        iou_thr_wbf=0.60,
        conf_type='max',
        skip_box_thr=0.001,
    )


# FETCH const.VOCAB_LONG FROM OUTPUT FILE #


def get_class_preds(preds_path_class: Path, preds_path_neg: Path, sname: str = 'test') -> DataFrame:
    ret = pd.read_csv(preds_path_class/f"{sname}.csv", index_col=0)
    ret.columns = ret.columns.map(const.VOCAB_SHORT_TO_LONG)
    neg_df = pd.read_csv(preds_path_neg/f"{sname}.csv", index_col=0)
    ret['2_class_positive'] = neg_df.positive
    ret['4_class_positive'] = 1 - ret.negative

    ret = ret.reset_index().rename(columns={'index': 'image_id'})

    img_to_study = preprocess.make_img_to_study_map(test_only=(sname == 'test'))
    ret['study_id'] = ret.image_id.map(img_to_study)

    study_positive = ret.groupby('study_id')[['2_class_positive', '4_class_positive']].mean().reset_index().rename(
        columns={'2_class_positive': '2_class_positive_study', '4_class_positive': '4_class_positive_study'}
    )
    ret = ret.merge(study_positive, on='study_id')

    return ret[['image_id', 'study_id', '2_class_positive_study', '4_class_positive_study'] + const.VOCAB_LONG]


# MAKE PREDICTIONS (unformatted) #


def get_box_preds_folds(
        preds_path_detect: Path, new_frame: Frame, sname: str = 'test', debug: bool = False
) -> Dict[str, List[Box]]:
    """
    DEBUG: Compute error
    """
    if debug:
        return train_detect.get_box_preds(preds_path_detect=preds_path_detect, new_frame=new_frame, sname=sname, fold=0)

    fold_preds = [
        train_detect.get_box_preds(preds_path_detect=preds_path_detect, new_frame=new_frame, sname=sname, fold=fold)
        for fold in range(5)
    ]
    return _combine_folds(fold_preds=fold_preds)


def format_class_pred_str(class_idx, conf) -> str:
    return f"{const.VOCAB_LONG[class_idx]} {conf} 0 0 1 1"


def format_df(ids, vals, typ: str) -> DataFrame:
    # typ: study or image
    df = DataFrame({'id': ids, 'PredictionString': vals})
    df.id = df.id + f"_{typ}"
    return df


def format_box(box: Box) -> str:
    assert box.C is not None
    xmin = box.X
    ymin = box.Y
    xmax = box.X + box.W
    ymax = box.Y + box.H
    return f"opacity {box.C} {xmin} {ymin} {xmax} {ymax}"


def _rescale(conf: float) -> float:
    """
    This is for testing purposes
    Rescales range 0 to 1 -> 0.9 to 0.1
    """
    return 0.9 + conf / 10


def format_none(conf: float) -> str:
    return f"none {conf} 0 0 1 1"


def make_study_predictions_hard(class_preds: DataFrame) -> DataFrame:
    data = class_preds.groupby('study_id')[const.VOCAB_LONG].mean()
    preds = data.to_numpy().argmax(1)
    preds_str = [format_class_pred_str(idx, 1.0) for idx in preds]

    id_str = data.reset_index().study_id
    return format_df(id_str, preds_str, typ="study")


def make_study_predictions_soft(class_preds: DataFrame) -> DataFrame:
    data = class_preds.groupby('study_id')[const.VOCAB_LONG].mean()
    preds_ls = [
        [format_class_pred_str(i, conf) for conf in data[cls]]
        for i, cls in enumerate(const.VOCAB_LONG)
    ]
    preds_str = list(map(' '.join, zip(*preds_ls)))

    id_str = data.reset_index().study_id
    return format_df(id_str, preds_str, typ="study")


def make_image_predictions_thresh(thresh, class_preds: DataFrame, box_preds: Dict[str, List[Box]]) -> DataFrame:
    id_to_str = {}
    for _i, row in class_preds.iterrows():
        id = row.image_id
        if row.study_positive < thresh:
            id_to_str[id] = format_none(1)
        else:
            id_to_str[id] = ' '.join(map(format_box, box_preds[id]))
    return format_df(ids=list(id_to_str.keys()), vals=list(id_to_str.values()), typ="image")


def _scale_box_confidences(boxes: List[Box], factor: float) -> List[Box]:
    assert all([b.C is not None for b in boxes])
    return [Box(X=b.X, Y=b.Y, W=b.W, H=b.H, C=b.C * factor) for b in boxes]


def make_image_predictions_combined(
        none_conf_base: float,
        class_positive_2_to_4: Tuple[float, bool],  # (amt, geom_mean)
        class_preds: DataFrame,
        box_preds: Dict[str, List[Box]],
) -> DataFrame:
    """
    class_positive_2_to_4: 0 = 2-class, 1 = 4-class, 0.5 = mean
    """
    id_to_str = {}
    for _i, row in class_preds.iterrows():
        id = row.image_id
        ratio_2_to_4, geom_mean = class_positive_2_to_4
        amt_2 = (1 - ratio_2_to_4)
        amt_4 = ratio_2_to_4
        if geom_mean:
            conf_pos = (row['2_class_positive_study'] ** amt_2) * (row['4_class_positive_study'] ** amt_4)
        else:
            conf_pos = row['2_class_positive_study'] * amt_2 + row['4_class_positive_study'] * amt_4
        conf_neg = 1.0 - conf_pos

        none_str = format_none(conf=none_conf_base * conf_neg)
        boxes = _scale_box_confidences(boxes=box_preds[id], factor=conf_pos)
        boxes_str = list(map(format_box, boxes))

        id_to_str[id] = ' '.join([none_str] + boxes_str)

    return format_df(ids=list(id_to_str.keys()), vals=list(id_to_str.values()), typ="image")


STUDY_PRED_HARD = make_study_predictions_hard
STUDY_PRED_SOFT = make_study_predictions_soft

IMAGE_PRED_THRESH = partial(make_image_predictions_thresh, thresh=0.5)
IMAGE_PRED_COMBINED_2 = partial(make_image_predictions_combined, none_conf_base=1.0, class_positive_2_to_4=(0., False))
IMAGE_PRED_COMBINED_2_GEOM = partial(make_image_predictions_combined, none_conf_base=1.0, class_positive_2_to_4=(0., True))
IMAGE_PRED_COMBINED_2_4 = partial(make_image_predictions_combined, none_conf_base=1.0, class_positive_2_to_4=(0.5, False))
IMAGE_PRED_COMBINED_2_4_GEOM = partial(make_image_predictions_combined, none_conf_base=1.0, class_positive_2_to_4=(0.5, True))
IMAGE_PRED_COMBINED_4 = partial(make_image_predictions_combined, none_conf_base=1.0, class_positive_2_to_4=(1., False))


def make_predictions(
        model_name_class_study: str,
        model_name_class_image: str,
        model_name_neg: str,
        model_name_detect: str,
        study_pred_func: Callable,
        image_pred_func: Callable,
        out_file_suffix: Optional[str] = None,
        sname: str = 'test',
        debug: bool = False,
) -> None:
    # Study predictions (class)
    if model_name_class_study.startswith("xgb"):
        preds_path_class_study = const.subdir_preds_xgb(path=True) / "study" / model_name_class_study
    else:
        preds_path_class_study = const.subdir_preds_class(path=True) / model_name_class_study

    # Image predictions (class / neg)
    if model_name_class_image.startswith("xgb"):
        preds_path_class_image = const.subdir_preds_xgb(path=True) / "image" / model_name_class_image
        preds_path_neg = const.subdir_preds_xgb(path=True) / "image" / model_name_neg
    else:
        preds_path_class_image = const.subdir_preds_class(path=True) / model_name_class_image
        preds_path_neg = const.subdir_preds_neg(path=True) / model_name_neg

    preds_path_detect = const.subdir_preds_detect(path=True) / model_name_detect

    out_name_parts = [model_name_class_study, model_name_class_image, model_name_neg, model_name_detect]
    if out_file_suffix is not None:
        out_name_parts.append(out_file_suffix)
    out_file = const.subdir_preds_final(path=True) / "__".join(out_name_parts)

    class_preds_study = get_class_preds(preds_path_class=preds_path_class_study, preds_path_neg=preds_path_neg, sname=sname)
    study_preds = study_pred_func(class_preds=class_preds_study)
    class_preds_image = get_class_preds(preds_path_class=preds_path_class_image, preds_path_neg=preds_path_neg, sname=sname)
    # Make sure that the frame here is whatever size is used by EfficientDet! That is the size at which it makes
    # its predictions
    box_preds = get_box_preds_folds(preds_path_detect=preds_path_detect, new_frame=Frame(256, 256), sname=sname, debug=debug)
    # ^ I spliced in this stuff here to get fast feedback on above code
    image_preds = image_pred_func(class_preds=class_preds_image, box_preds=box_preds)


    preds = pd.concat([study_preds, image_preds])
    preds.to_csv(f"{out_file}.csv", index=False)


def test():
    make_predictions(
        model_name_class_study="resnet18",
        model_name_class_image="resnet18",
        model_name_neg="resnet18",
        model_name_detect="eff_lite0_256",
        study_pred_func=STUDY_PRED_SOFT,
        image_pred_func=IMAGE_PRED_COMBINED_2_4,
        sname='valid',
    )

    make_predictions(
        model_name_class_study="xgb_5fold",
        model_name_class_image="xgb_5fold_class",
        model_name_neg="xgb_5fold_neg",
        model_name_detect="eff_lite0_256",
        study_pred_func=STUDY_PRED_SOFT,
        image_pred_func=IMAGE_PRED_COMBINED_2_4,
        sname='test',
    )