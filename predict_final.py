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

import const
import preprocess
from make_data_detect import Frame, Box
from train_detect import get_box_preds


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
        model_name_class: str,
        model_name_neg: str,
        model_name_detect: str,
        study_pred_func: Callable,
        image_pred_func: Callable,
        out_file_suffix: Optional[str] = None,
        sname: str = 'test',
) -> None:
    preds_path_class = const.subdir_preds_class(path=True) / model_name_class
    preds_path_neg = const.subdir_preds_neg(path=True) / model_name_neg
    preds_path_detect = const.subdir_preds_detect(path=True) / model_name_detect

    out_name_parts = [model_name_class, model_name_neg, model_name_detect]
    if out_file_suffix is not None:
        out_name_parts.append(out_file_suffix)
    out_file = const.subdir_preds_final(path=True) / "__".join(out_name_parts)

    class_preds = get_class_preds(preds_path_class=preds_path_class, preds_path_neg=preds_path_neg, sname=sname)
    # Make sure that the frame here is whatever size is used by EfficientDet! That is the size at which it makes
    # its predictions
    box_preds = get_box_preds(preds_path_detect=preds_path_detect, new_frame=Frame(256, 256), sname=sname)

    study_preds = study_pred_func(class_preds=class_preds)
    image_preds = image_pred_func(class_preds=class_preds, box_preds=box_preds)

    preds = pd.concat([study_preds, image_preds])
    preds.to_csv(f"{out_file}.csv", index=False)


def test():
    make_predictions(
        model_name_class="resnet18",
        model_name_neg="resnet18",
        model_name_detect="eff_lite0_256",
        study_pred_func=STUDY_PRED_SOFT,
        image_pred_func=IMAGE_PRED_COMBINED_2_4,
        sname='valid',
    )