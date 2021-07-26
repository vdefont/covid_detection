from pathlib import Path
import shutil
from typing import Tuple, List, Dict, Optional, Set, Iterable, Any, NamedTuple
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
import numpy as np
import itertools
import pickle
from PIL import Image, ImageDraw
import json
from multiprocessing import Pool

import const
import utils


# MAKING MASKS #


class MakeMaskArgs(NamedTuple):
    base_dir: Path
    row: Any
    size: int
    extn: str


def _make_mask(args: MakeMaskArgs) -> None:
    row = args.row
    mask = Image.new('L', size=(row.width, row.height))
    if isinstance(row.boxes, str):
        for box in json.loads(row.boxes):
            draw = ImageDraw.Draw(mask)
            x1 = box["x"]
            y1 = box["y"]
            x2 = box["x"] + box["width"]
            y2 = box["y"] + box["height"]
            draw.rectangle([(x1, y1), (x2, y2)], fill="#ffffff")

    mask = utils.resize(array=np.array(mask), size=args.size)
    mask.save(args.base_dir / f"{row.id}.{args.extn}")


def make_masks(size: int = 256, extn: str = "png") -> None:
    """
    Run this once to create jpg/png images of the masks
    Takes 2 minutes to run
    """
    base_dir = Path(const.subdir_data_class(path=True) / 'masks')
    base_dir.mkdir()

    data = pd.read_csv(const.subdir_data_csv(path=True) / "train_image_level_prep.csv")
    args = [MakeMaskArgs(base_dir=base_dir, row=row, size=size, extn=extn) for _i, row in data.iterrows()]
    pool = Pool(processes=5)
    pool.map(_make_mask, args)


# MAIN STUFF #


def get_tr_vl(data: Optional[DataFrame] = None, valid_amt: float = 0.3) -> Tuple[Set[str], Set[str]]:
    """
    Strategy:
    - Group images by study_id, and bucket these groups by label
    - For each label bucket, split up the gps so that train+val
      have about an equal number of individual samples
    """
    data = data or utils.setup_data()

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


def _get_fold_index(folds: List[Set[str]], val: str) -> int:
    for i, fold in enumerate(folds):
        if val in fold:
            return i
    raise Exception(f"{val} not found in any of the {len(folds)} folds!")


def populate_dirs_folds(data: DataFrame, folds: List[Set[str]], src: Path, extn: str, dst: Path) -> None:
    # Set up dirs
    for c, fold_i in itertools.product(const.VOCAB_SHORT, range(len(folds))):
        (dst / f"fold{fold_i}" / c).mkdir(parents=True)

    # Copy images
    img_id_to_label = dict(zip(data.img_id, data.label))
    for p in (src / 'train').glob(f'*{extn}'):
        id = p.name[:-4]
        lab = img_id_to_label[id]
        fold_i = _get_fold_index(folds=folds, val=id)
        shutil.copy(p, dst/f"fold{fold_i}"/lab/p.name)



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
    data = utils.setup_data()
    X_tr, X_ts = get_tr_vl(data=data, valid_amt=valid_amt)
    populate_dirs(data=data, X_tr=X_tr, X_ts=X_ts, src=src, extn=extn, dst=dst)


def create_data_folds(src: Path, extn: str, dst: Path, num_folds: int) -> None:
    data = utils.setup_data()
    folds = utils.get_folds(data=data, num_folds=num_folds)
    populate_dirs_folds(data=data, folds=folds, src=src, extn=extn, dst=dst)


def create_data_test_only(src: Path, dst: Path) -> None:
    for lab, sname in itertools.product(const.VOCAB_SHORT, ['train', 'test']):
        d = dst/sname/lab
        if not d.exists():
            d.mkdir(parents=True)

    vocab_cycle = itertools.cycle(const.VOCAB_SHORT)
    for i, (p, lab) in enumerate(zip((src/'test').glob('*'), vocab_cycle)):
        shutil.copy(p, dst/'test'/lab/p.name)
        # Also copy some images to "train"
        # This is required for our dataloader to work
        if i < 70:
            shutil.copy(p, dst / 'train' / lab / p.name)


def validate_created_data(extn: str, dst: Path, test_only: bool) -> None:
    def num_imgs(p):
        return len(list(p.glob(f'**/*{extn}')))

    def vocab_amts(p):
        for c in const.VOCAB_SHORT:
            print(f"{c}: {num_imgs(p / c)}")
        print(f"Total: {num_imgs(p)}")

    for dir_p in dst.glob('*'):
        if not dir_p.is_dir():
            continue
        print(dir_p.name)
        vocab_amts(dir_p)
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


def create_and_validate_data_folds(src: str, dst: str, num_folds: int, extn: str = "png") -> None:
    src_path = const.subdir_data_image(path=True) / src
    dst_path = const.subdir_data_class(path=True) / dst
    create_data_folds(src=src_path, extn=extn, dst=dst_path, num_folds=num_folds)
    validate_created_data(extn=extn, dst=dst_path, test_only=False)


def create_and_validate_data_mf(src: str, test_only: bool = False) -> None:
    """
    Given an input directory "png224", makes two new directories titled
    "png224_m" and "png224_f"
    """
    src_path = const.subdir_data_class(path=True) / src

    # Get male/female info
    meta = pd.read_csv(const.subdir_data_csv() + "metadata_feats_train.csv")
    males = set(meta.image_id[meta.PatientSex_M == 1])

    # Set up dirs
    dir_m = Path(str(src_path) + "_m")
    dir_f = Path(str(src_path) + "_f")
    snames = ['train', 'test'] if test_only else ['train', 'valid']
    for dir_base, sname, label in itertools.product([dir_m, dir_f], snames, const.VOCAB_SHORT):
        (dir_base/sname/label).mkdir(parents=True)

    # Copy over files
    for p in src_path.glob('**/*png'):
        img_id = p.name.replace(".png", "")
        dir_dst = dir_m if img_id in males else dir_f
        dst = dir_dst / p.parent.parent.name / p.parent.name / p.name
        shutil.copy(p, dst)

    print("MALES:\n")
    validate_created_data(extn="png", dst=dir_m, test_only=test_only)
    print("FEMALES:\n")
    validate_created_data(extn="png", dst=dir_f, test_only=test_only)


# create_and_validate_data_folds(src="png224", dst="png224_3fold", num_folds=3)
# create_and_validate_data_folds(src="jpg224", dst="jpg224_3fold", num_folds=3, extn="jpg")
# create_and_validate_data_folds(src="jpg768", dst="jpg768_5fold", num_folds=5, extn="jpg")
# create_and_validate_data_folds(src="jpg768", dst="jpg768_3fold", num_folds=3, extn="jpg")
# create_and_validate_data_folds(src="jpg640", dst="jpg640_3fold", num_folds=3, extn="jpg")
# create_and_validate_data_folds(src="jpg640", dst="jpg640_5fold", num_folds=5, extn="jpg")
# create_and_validate_data_folds(src="jpg1024", dst="jpg1024_3fold", num_folds=3, extn="jpg")
# create_and_validate_data_folds(src="png512", dst="png512_5fold", num_folds=5, extn="png")
# create_and_validate_data_folds(src="png512", dst="png512_3fold", num_folds=3, extn="png")
# create_and_validate_data_folds(src="png224", dst="png224_5fold", num_folds=5)
# create_and_validate_data_folds(src="png224", dst="png224_10fold", num_folds=10)
# create_and_validate_data(src="png224", extn="png", dst="png224", valid_amt=0.3)