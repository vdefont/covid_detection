from pathlib import Path
import shutil
from typing import Tuple, List, Dict, Optional, Set, Iterable
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import numpy as np
import itertools
import pickle
from PIL import Image

import const


def _make_square(im: np.ndarray) -> np.ndarray:
    """
    Pads the left/right or top/bottom of the image with black, so that
    it is a square
    """
    dim = max(im.shape)
    ret = np.zeros((dim, dim)).astype(np.uint8)
    y_start = (dim - im.shape[0]) // 2
    x_start = (dim - im.shape[1]) // 2
    ret[y_start: y_start + im.shape[0], x_start: x_start + im.shape[1]] = im
    return ret


def resize(array, size, resample=Image.LANCZOS):
    # Forked from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    array = _make_square(array)  # Ensures that resizing preserves the dimensions
    im = Image.fromarray(array)

    im = im.resize((size, size), resample)

    return im


def setup_data() -> DataFrame:
    """
    Combine image and study CSV files, remove duplicates
    """
    img_data = pd.read_csv(const.DIR_ORIGINAL_DATA + "train_image_level.csv")
    img_data = img_data.rename(columns={'id': 'img_id', 'StudyInstanceUID': 'study_id'})
    img_data.img_id = img_data.img_id.str[:-len("_image")]

    # Remove dups
    with open("data_original/dups", "rb") as f:
        dups = pickle.load(f)
    img_data = img_data[~img_data.img_id.isin(dups)]


    study_data = pd.read_csv(const.DIR_ORIGINAL_DATA + "train_study_level.csv")
    study_data = study_data.rename(columns={'id': 'study_id'})
    study_data.study_id = study_data.study_id.str[:-len("_study")]
    study_data.columns = ["study_id"] + const.VOCAB_SHORT

    data = img_data.merge(study_data, on='study_id')[['img_id', 'study_id']+const.VOCAB_SHORT]

    # Add labels
    d = data[const.VOCAB_SHORT].stack().reset_index()
    data['label'] = d[d[0] == 1]['level_1'].tolist()
    return data


def get_folds(
        data: Optional[DataFrame] = None,
        num_folds: int = 5,
        boxes: bool = False,
        seed: int = 42,
) -> List[Set[str]]:
    """
    Strategy:
    - Group images by study_id, and bucket these groups by label
        (so we have four buckets, each containing many groups)
    - Split up the groups so that the five folds have about an
     equal number of individual samples

    If boxes is True, we only consider images where there are boxes
    """
    data = _setup_data() if data is None else data

    if boxes is True:
        image_data = pd.read_csv(const.subdir_data_csv(path=True) / "train_image_level_prep.csv")
        ids_with_boxes = image_data.id[~image_data.boxes.isna()]
        data = data[data.img_id.isin(ids_with_boxes)]

    random.seed(seed)
    lab_to_gps = defaultdict(list)
    for name, group in data.groupby(['label', 'study_id']):
        lab_to_gps[name[0]].append(group)

    fold_gps = [[] for _ in range(num_folds)]
    for gps in lab_to_gps.values():
        random.shuffle(gps)
        cumsum = np.cumsum([len(gp) for gp in gps])
        cum_frac = cumsum / cumsum[-1] - 0.000001 # So the last one is not in next bucket
        fold_i_ls = (cum_frac * num_folds).astype(int)
        for fold_i, gp in zip(fold_i_ls, gps):
            fold_gps[fold_i].append(gp)

    return [set(pd.concat(gps).img_id) for gps in fold_gps]