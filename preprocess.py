import pandas as pd
from pandas import DataFrame
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from copy import deepcopy
import pickle
from pathlib import Path
from typing import List, Dict, Optional, NamedTuple, Tuple, Any, Union
from multiprocessing import Pool
import os
from PIL import Image
from tqdm.auto import tqdm
import random

import const


# MAKE IMAGES #


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im


class MakeImageArgs(NamedTuple):
    dirname: str
    file: str
    extn: str
    size: int
    save_dir: Path


def make_image(args: MakeImageArgs) -> Tuple[int]:
    """
    Makes image, and returns shape of image created
    """

    # set keep_ratio=True to have original aspect ratio
    xray = read_xray(os.path.join(args.dirname, args.file))
    im = resize(xray, size=args.size)
    im.save(args.save_dir/args.file.replace('dcm', args.extn))

    return xray.shape


def make_images(extn: str, size: int, dst: Optional[str] = None, test_only: bool = False) -> None:
    dst = dst or f'{extn}{size}'

    image_id = []
    splits = []

    make_image_args: List[MakeImageArgs] = []

    # Need to also have some train so that our dataloaders works. But we only take 70
    # so that there is enough for one batch
    for split in ['test'] if test_only else ['train']:
        save_dir = const.subdir_data_image(path=True) / f'{dst}/{split}'

        save_dir.mkdir(exist_ok=False, parents=True)
        print("CREATED:", save_dir)

        for dirname, _, filenames in tqdm(os.walk(const.DIR_ORIGINAL_DATA + split)):
            for file in filenames:
                make_image_args.append(
                    MakeImageArgs(dirname=dirname, file=file, extn=extn, size=size, save_dir=save_dir)
                )

                image_id.append(file[:-len('.dcm')])
                splits.append(split)
                break # TODO DELETE
            break # TODO delete

    pool = Pool(processes=5)
    shapes = pool.map(make_image, make_image_args)

    meta_csv_out = const.subdir_data_csv(path=True) / ("meta_test.csv" if test_only else "meta.csv")
    if not meta_csv_out.exists():
        print("MAKING ", meta_csv_out)
        if not meta_csv_out.parent.exists():
            print("DOES NOT EXIST: Creating", meta_csv_out.parent)
            meta_csv_out.parent.mkdir()
        else:
            print("DOES EXIST!")
        dim0, dim1 = tuple(zip(*shapes))
        df = DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1, 'split': splits})
        df.to_csv(meta_csv_out, index=False)


# MAKE CSV #


def _preprocess_img_csv(data: DataFrame) -> DataFrame:
    data = deepcopy(data)

    # Make the "id" field cleaner
    data.id = data.id.str.replace("_image", "")

    # Add in original width + height
    meta = (
        pd.read_csv(const.DIR_WORKING + const.SUBDIR_DATA_CSV + "meta.csv").rename(columns={'dim0': 'height', 'dim1': 'width'})
    )
    data = data.merge(meta, left_on='id', right_on='image_id')

    # Allow JSON parsing of this field
    data.boxes = data.boxes.str.replace("'", '"')

    return data[['id', 'boxes', 'label', 'height', 'width']]


def _preprocess_study_csv(data: DataFrame) -> DataFrame:
    data = deepcopy(data)

    # Make the "id" field cleaner
    data.id = data.id.str.replace("_study", "")

    return data


def preprocess_csv():
    img_data = pd.read_csv(const.dir_original_data() + "train_image_level.csv")
    img_data_prep = _preprocess_img_csv(img_data)
    img_data_prep.to_csv(const.subdir_data_csv() + "train_image_level_prep.csv", index=False)

    study_data = pd.read_csv(const.DIR_ORIGINAL_DATA + "train_study_level.csv")
    study_data_prep = _preprocess_study_csv(study_data)
    study_data_prep.to_csv(const.subdir_data_csv() + "train_study_level_prep.csv", index=False)


def make_id_orig_size_csv(test_only: bool = False):
    meta_path = "meta_test.csv" if test_only else "meta.csv"
    data = pd.read_csv(const.DIR_WORKING + const.SUBDIR_DATA_CSV + meta_path)
    data = data.rename(columns={'image_id': 'id', 'dim0': 'height', 'dim1': 'width'})
    data = data[['id', 'width', 'height']]
    out_path = 'id_orig_size_test.csv' if test_only else 'id_orig_size.csv'
    data.to_csv(const.DIR_WORKING + const.SUBDIR_DATA_CSV + out_path, index=False)


def make_img_to_study_map(test_only: bool = False) -> Dict[str, str]:
    """
    Scan through original images and make a map img->study
    """
    ret = {}
    for sname in ['test'] if test_only else ["train", "test"]:
        for p in Path(const.DIR_ORIGINAL_DATA + sname).glob('**/*dcm'):
            study = p.parent.parent.name
            img = p.name[:-len(".dcm")]
            ret[img] = study
    return ret


# ADD METADATA #


DCM_PROPS = [
    "SpecificCharacterSet", "ImageType", "Modality", "PatientSex", "BodyPartExamined",
    "PhotometricInterpretation", "Rows", "Columns", "SeriesNumber", "InstanceNumber",
    "SamplesPerPixel",
]


def get_row(path) -> Dict[str, Any]:
    row = {}
    row["image_id"] = path.name.replace(".dcm", "")
    row["series_id"] = path.parent.name
    row["study_id"] = path.parent.parent.name

    dcm = pydicom.dcmread(path)
    for prop in DCM_PROPS:
        row[prop] = dcm[prop].value

    return row


def get_metadata_raw(sname):
    pool = Pool()
    rows = pool.map(get_row, (const.dir_original_data(path=True)/sname).glob('**/*dcm'))
    data = DataFrame(rows)

    # Add field "images_in_study"
    imgs_per_study = data.groupby("study_id").image_id.count().reset_index().rename(
        columns={'image_id': 'images_in_study'})
    data = data.merge(imgs_per_study, on="study_id")

    return data


# Adding generic features
def add_feats(data: DataFrame, feats: DataFrame, feat: str, vals: Optional[List[Union[str, int]]] = None) -> None:
    vals = vals or data[feat].unique()

    for val in vals:
        feats[f"{feat}_{val}"] = data[feat] == val


def test_add_feats(data):
    feats = data[[]]
    add_feats(data, feats, "PatientSex")
    return feats.sum(0)


def add_feats_images_in_study(data: DataFrame, feats: DataFrame) -> None:
    feats['images_in_study'] = data.images_in_study
    # Cap it at 3
    feats.images_in_study[feats.images_in_study >= 3] = 3


def add_feats_series_number(data: DataFrame, feats: DataFrame) -> None:
    # This one is categorical
    add_feats(data, feats, "SeriesNumber", [1, 2, 3])
    feats["SeriesNumber_4+"] = (data.SeriesNumber >= 4) & (data.SeriesNumber < 15)
    feats["SeriesNumber_1000"] = data.SeriesNumber >= 1000


def add_feats_instance_number(data: DataFrame, feats: DataFrame) -> None:
    # This one is categorical
    add_feats(data, feats, "InstanceNumber", [1, 2])
    feats["InstanceNumber_3+"] = (data.InstanceNumber >= 3) & (data.InstanceNumber <= 15)
    feats["InstanceNumber_1000"] = data.InstanceNumber >= 1000


def add_feats_image_type(data: DataFrame, feats: DataFrame) -> None:
    data["ImageTypeStr"] = data.ImageType.astype(str)
    for feat in ["ORIGINAL", "PRIMARY", "DERIVED", "SECONDARY", "CSA RESAMPLED",
                 "POST_PROCESSED", "''", "RT", "100000"]:
        feats[f'image_type_{feat}'] = data.ImageTypeStr.str.contains(feat)


def test_add_feats_image_type(data):
    data2 = add_feats_image_type(data)
    for feat in ["ORIGINAL", "PRIMARY", "DERIVED", "SECONDARY", "CSA RESAMPLED",
                "POST_PROCESSED", "''", "RT", "100000"]:
        print(f"{feat} \t {data2[f'image_type_{feat}'].sum():.4f}")


def add_feats_body_part(data: DataFrame, feats: DataFrame) -> None:
    bpe = data.BodyPartExamined

    feats['part_chest'] = bpe == 'CHEST'
    feats['part_'] = bpe == ''
    feats['part_torax'] = bpe.isin(['TORAX', 'THORAX', '2- TORAX'])
    feats['part_port_chest'] = bpe == 'PORT CHEST'
    # We use oo instead of accented o
    feats['part_toorax'] = bpe.isin(['T?RAX', 'TÒRAX'])
    feats['part_skull'] = bpe == 'SKULL'
    feats['part_abdomen'] = bpe == 'ABDOMEN'
    feats['part_pecho'] = bpe.isin(['Pecho', 'PECHO'])

    feats['part_gp_chest'] = bpe.isin(
        ['CHEST', 'PORT CHEST', 'Pecho', 'PECHO']
    )
    feats['part_gp_thorax'] = bpe.isin(
        ['TORAX', 'T?RAX', 'THORAX', '2- TORAX', 'TÒRAX']
    )
    feats['part_gp_mexico'] = bpe.isin(
        ['T?RAX', 'TÒRAX', 'Pecho', 'PECHO']
    )


def add_feats_sex(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data=data, feats=feats, feat="PatientSex")


def add_feats_photometric(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "PhotometricInterpretation")


def add_feats_modality(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "Modality")


def add_feats_charset(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "SpecificCharacterSet")


def make_metadata_feats(sname: str) -> None:
    # Get raw metadata
    data = get_metadata_raw(sname=sname)

    # Convert to feats
    feats = deepcopy(data)[["study_id", "image_id"]]
    to_add = [
        add_feats_images_in_study,
        add_feats_series_number,
        add_feats_instance_number,
        add_feats_image_type,
        add_feats_body_part,
        add_feats_sex,
        add_feats_photometric,
        add_feats_modality,
        add_feats_charset,
    ]
    for add in to_add:
        add(data=data, feats=feats)

    # Clean up and return
    feats = feats.set_index(["study_id", "image_id"])
    feats = feats.astype(int)  # All our feats should be int

    feats.to_csv(const.subdir_data_csv(path=True)/f'metadata_feats_{sname}.csv')


def verify_feats(feats):
    for feat in feats:
        print(feat, sum(feats[feat]))