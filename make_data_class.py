from pathlib import Path
import shutil
from typing import Tuple, List, Dict, Optional, Set, Iterable
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import numpy as np

import const


def setup_data() -> DataFrame:
    img_data = pd.read_csv(const.DIR_ORIGINAL_DATA + "train_image_level.csv")
    img_data = img_data.rename(columns={'id': 'img_id', 'StudyInstanceUID': 'study_id'})
    img_data.img_id = img_data.img_id.str[:-6]

    study_data = pd.read_csv(const.DIR_ORIGINAL_DATA + "train_study_level.csv")
    study_data = study_data.rename(columns={'id': 'study_id'})
    study_data.study_id = study_data.study_id.str[:-6]
    study_data.columns = ["study_id"] + const.VOCAB_SHORT

    data = img_data.merge(study_data, on='study_id')[['img_id', 'study_id']+const.VOCAB_SHORT]

    # Add labels
    d = data[const.VOCAB_SHORT].stack().reset_index()
    data['label'] = d[d[0] == 1]['level_1'].tolist()
    return data


def get_tr_vl(data: DataFrame, valid_amt: float) -> Tuple[Set[str], Set[str]]:
    """
    Strategy:
    - Group images by study_id, and bucket these groups by label
    - For each label bucket, split up the gps so that train+val
      have about an equal number of individual samples
    """
    random.seed(42)
    lab_to_gps = defaultdict(list)
    for name, group in data.groupby(['label', 'study_id']):
        lab_to_gps[name[0]].append(group)

    tr_gps = []
    vl_gps = []
    for gps in lab_to_gps.values():
        random.shuffle(gps)
        cumsum = np.cumsum(list(map(len, gps)))
        num = cumsum[-1]
        tr_vl_spl = (cumsum >= (1-valid_amt)*num).nonzero()[0][0]
        tr_gps += gps[:tr_vl_spl]
        vl_gps += gps[tr_vl_spl:]

    tr = pd.concat(tr_gps).img_id
    vl = pd.concat(vl_gps).img_id
    return set(tr), set(vl)


def populate_dirs(data: DataFrame, X_tr: Iterable[str], X_ts: Iterable[str], src: Path, extn: str, dst: Path) -> None:
    # Set up dirs
    for c in const.VOCAB_SHORT:
        (dst/'train'/c).mkdir(parents=True)
        (dst/'valid'/c).mkdir(parents=True)


    # Copy images
    img_id_to_label = dict(zip(data.img_id, data.label))
    for p in (src/'train').glob(f'*{extn}'):
        id = p.name[:-4]
        lab = img_id_to_label[id]
        is_tr = id in X_tr
        is_ts = id in X_ts
        assert is_tr ^ is_ts
        sname = 'train' if is_tr else 'valid'
        shutil.copy(p, dst/sname/lab/p.name)


def create_data(src: Path, extn: str, dst: Path, valid_amt: float) -> None:
    data = setup_data()
    X_tr, X_ts = get_tr_vl(data=data, valid_amt=valid_amt)
    populate_dirs(data=data, X_tr=X_tr, X_ts=X_ts, src=src, extn=extn, dst=dst)


def create_data_test_only(src: Path, dst: Path) -> None:
    shutil.copytree(src/'test', dst/'test'/'neg')

    # Also copy some images to "train"
    # This is required for our dataloader to work
    (dst/'train'/'neg').mkdir(parents=True)
    for i, p in enumerate((dst/'test'/'neg').glob('*')):
        if i == 70:
            break
        shutil.copy(p, dst/'train'/'neg'/p.name)


def validate_created_data(extn: str, dst: Path, test_only: bool) -> None:
    def num_imgs(p):
        return len(list(p.glob(f'**/*{extn}')))

    def vocab_amts(p):
        for c in const.VOCAB_SHORT:
            print(f"{c}: {num_imgs(p / c)}")
        print(f"Total: {num_imgs(p)}")

    for sname in ['train', 'test'] if test_only else const.SNAMES:
        print(sname)
        vocab_amts(dst/sname)
        print()


def create_and_validate_data(
        src: str, dst: str, extn: Optional[str] = None, valid_amt: Optional[float] = None, test_only: bool = False
) -> None:
    src_path = const.subdir_data_image(path=True) / src
    dst_path = const.subdir_data_class(path=True) / dst

    if test_only:
        create_data_test_only(src=src_path, dst=dst_path)
    else:
        assert extn is not None
        assert valid_amt is not None
        create_data(src=src_path, extn=extn, dst=dst_path, valid_amt=valid_amt)

    validate_created_data(extn=extn or 'png', dst=dst_path, test_only=test_only)


# create_and_validate_data(src="png224", extn="png", dst="png224", valid_amt=0.3)