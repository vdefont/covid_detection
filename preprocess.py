from copy import deepcopy
import pandas as pd
from pandas import DataFrame
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from copy import deepcopy
import pickle
from pathlib import Path
from typing import List, Dict, Optional, NamedTuple, Tuple, Any, Union, Set
from multiprocessing import Pool
import os
from PIL import Image
from tqdm.auto import tqdm
import random

import const
import utils


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


def see_big_ratios() -> DataFrame:
    meta = pd.read_csv("data_csv/meta.csv")
    meta['r01'] = meta.dim0 / meta.dim1
    meta['r10'] = meta.dim1 / meta.dim0
    meta[['r01', 'r10']].max(axis=1)
    meta['r'] = meta[['r01', 'r10']].max(axis=1)
    meta = meta.sort_values(by='r', ascending=False)
    # meta.r.plot.hist(bins=20)
    # - Clearly shows that we should be resizing better!
    return meta


def show_dcm(name: str) -> None:
    """
    For exploratory purposes. Check out:
    - a5c5e8425f03
    """
    for p in Path("data_original/train").glob('**/*dcm'):
        if p.name.replace(".dcm", "") == name:
            im = read_xray(p)
            break
    img = Image.fromarray(im)
    img.show()


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
    im = utils.resize(xray, size=args.size)
    im.save(args.save_dir / args.file.replace('dcm', args.extn))

    return xray.shape


def make_images(extn: str, size: int, dst: Optional[str] = None, test_only: bool = False,
                cut_1263: bool = True) -> None:
    dst = dst or f'{extn}{size}'

    image_id = []
    splits = []

    make_image_args: List[MakeImageArgs] = []

    dups = get_dups()

    # Need to also have some train so that our dataloaders works. But we only take 70
    # so that there is enough for one batch
    for split in ['test'] if test_only else ['train']:
        save_dir = const.subdir_data_image(path=True) / f'{dst}/{split}'

        save_dir.mkdir(exist_ok=False, parents=True)
        print("CREATED:", save_dir)

        num = 0
        total = len(list(Path(const.dir_original_data(path=True) / split).glob('**/*dcm')))
        for dirname, _, filenames in tqdm(os.walk(const.DIR_ORIGINAL_DATA + split)):
            for file in filenames:
                # Kaggle dummy test set - skip over
                if cut_1263 and total == 1263 and num >= 70:
                    continue
                if not file.endswith('.dcm'):
                    continue
                image_id_i = file[:-len('.dcm')]
                if image_id_i in dups:
                    continue

                num += 1

                make_image_args.append(
                    MakeImageArgs(dirname=dirname, file=file, extn=extn, size=size, save_dir=save_dir)
                )

                image_id.append(image_id_i)
                splits.append(split)

    pool = Pool(processes=5)
    shapes = pool.map(make_image, make_image_args)

    meta_csv_out = const.subdir_data_csv(path=True) / ("meta_test.csv" if test_only else "meta.csv")
    if not meta_csv_out.exists():
        if not meta_csv_out.parent.exists():
            meta_csv_out.parent.mkdir()
        dim0, dim1 = tuple(zip(*shapes))
        df = DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1, 'split': splits})
        df.to_csv(meta_csv_out, index=False)


# FIND DUPLICATES TO REMOVE #


def _mkarr(p: str) -> np.array:
    return np.array(Image.open(f"data_image/png224_dups/train/{p}.png"))


def _is_same(p1: str, p2: str) -> bool:
    na, nb = _mkarr(p1), _mkarr(p2)
    return (na == nb).sum((0, 1)) / (224 ** 2) > 0.99


def find_and_save_dups():
    """
    Requires:
    - created images with dups
    - created train_image_level_prep
    """

    # duplicate -> original
    dups = {}

    data = pd.read_csv("data_csv/train_image_level_prep.csv")
    data["study"] = data.id.map(make_img_to_study_map())
    for _study, group in data.groupby("study"):
        id_to_boxes = dict(zip(group.id, group.boxes))

        # Process each image to sort into unique groups
        unique_gps = []
        for img_id in group.id:
            found = False
            # See if it's in an existing group
            for ug in unique_gps:
                if not found and any(_is_same(img_id, ug_id) for ug_id in ug):
                    ug.add(img_id)
                    found = True
            # If not, make a new group with it
            if not found:
                unique_gps.append({img_id})

        for gp in unique_gps:
            # Determine what to keep
            boxes = {np.nan}
            keep = set()
            for id in gp:
                boxes_i = id_to_boxes[id]
                if boxes_i not in boxes:
                    boxes.add(boxes_i)
                    keep.add(id)
            if len(keep) == 0:
                keep = {list(gp)[0]}

            # Mark duplicates
            orig = list(keep)[0]
            for dup in gp - keep:
                dups[dup] = orig

    with open("data_original/dups", "wb") as f:
        pickle.dump(set(dups), f)


def get_dups() -> Set[str]:
    p = Path("data_original/dups")
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return set()


# MAKE CSV #


def _preprocess_img_csv(data: DataFrame) -> DataFrame:
    data = deepcopy(data)

    # Make the "id" field cleaner
    data.id = data.id.str.replace("_image", "")

    # Add in original width + height
    meta = (
        pd.read_csv(const.DIR_WORKING + const.SUBDIR_DATA_CSV + "meta.csv").rename(
            columns={'dim0': 'height', 'dim1': 'width'})
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
    "SamplesPerPixel", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "DeidentificationMethod",
]

META_PROPS = ["MediaStorageSOPClassUID", "TransferSyntaxUID"]

TAGS = {
    "TAG_0029_0010": (0x0029, 0x0010),
    "TAG_private_creator_geiss_pacs": (0x0903, 0x0010),
    "TAG_private_creator_geiss": (0x0905, 0x0010),
    "TAG_2001_0010": (0x2001, 0x0010),
    "TAG_200B_0010": (0x200B, 0x0010),
    "TAG_200B_0011": (0x200B, 0x0011),
    "TAG_200B_0012": (0x200B, 0x0012),
    "TAG_7FD1_0010": (0x7FD1, 0x0010),
}


def get_row(path) -> Dict[str, Any]:
    row = {}
    row["image_id"] = path.name.replace(".dcm", "")
    row["series_id"] = path.parent.name
    row["study_id"] = path.parent.parent.name

    dcm = pydicom.dcmread(path)
    dcm._dict
    for prop in DCM_PROPS:
        row[prop] = dcm[prop].value
    row["ImagerPixelSpacing"] = dcm["ImagerPixelSpacing"].value[0]  # The x and y spacing are the same

    for meta_prop in META_PROPS:
        row[meta_prop] = str(dcm.file_meta[meta_prop]).split('    ')[-1].lstrip()

    for tag_name, tag_val in TAGS.items():
        row[tag_name] = int(tag_val in dcm)

    return row


def get_metadata_raw(sname):
    pool = Pool()
    rows = pool.map(get_row, (const.dir_original_data(path=True) / sname).glob('**/*dcm'))
    data = DataFrame(rows)

    # Add field "images_in_study"
    imgs_per_study = data.groupby("study_id").image_id.count().reset_index().rename(
        columns={'image_id': 'images_in_study'})
    data = data.merge(imgs_per_study, on="study_id")

    return data


# Adding generic features
def add_feats(
        data: DataFrame,
        feats: DataFrame,
        feat: str,
        vals: Optional[List[Union[str, int]]] = None,
        verbose: bool = False
) -> None:
    if vals is None:
        raise Exception("Please provide vals! (comment this for testing)")
    vals = vals or data[feat].unique()
    if verbose:
        print(f"For feat {feat}, adding vals: {vals}")

    for val in vals:
        feats[f"{feat}_{val}"] = data[feat] == val


def test_add_feats(data):
    feats = data[[]]
    add_feats(data, feats, "PatientSex")
    return feats.sum(0)


def add_feats_rows_cols(data: DataFrame, feats: DataFrame) -> None:
    feats['rows'] = data.Rows
    feats['cols'] = data.Columns
    feats['area'] = feats.rows * feats.cols
    feats['aspect_ratio'] = feats.rows / feats.cols


def add_feats_pixel_spacing(data: DataFrame, feats: DataFrame) -> None:
    feats['PixelSpacing'] = data.ImagerPixelSpacing
    # Make as categorical variables
    print(f"OLD CHOICES: {data.ImagerPixelSpacing.unique()}")
    print(f"TYPE: {type(data.ImagerPixelSpacing.unique()[0])}")
    for spacing in [
        0.148, 0.139, 0.1, 0.15, 0.14, 0.138999998569489, 0.125, 0.175, 0.308553,
        0.143, 0.2, 0.0875, 0.1988, 0.2000000029802, 0.168, 0.144, 0.194222, 0.160003,
        0.1868, 0.194549, 0.194553, 0.194311, 0.194556, 0.187667, 0.1852, 0.1902
    ]:
        feats[f'PixelSpacing_{spacing}'] = feats.PixelSpacing == spacing
    feats['PixelSpacing_0.2_all'] = feats['PixelSpacing_0.2'] | feats['PixelSpacing_0.2000000029802']
    feats['PixelSpacing_0.139_all'] = feats['PixelSpacing_0.138999998569489'] | feats['PixelSpacing_0.139']
    # Make as numerical variable
    feats['PixelSpacing'] = feats.PixelSpacing.astype(float)


def add_feats_row_col_pixel_spacing(data: DataFrame, feats: DataFrame) -> None:
    spacing = data.ImagerPixelSpacing
    rows = data.Rows
    cols = data.Columns
    scaled_rows = rows * spacing
    scaled_cols = cols * spacing
    scaled_area = scaled_rows * scaled_cols
    feats['scaled_rows'] = scaled_rows
    feats['scaled_cols'] = scaled_cols
    feats['scaled_area'] = scaled_area


def add_feats_bits(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, 'BitsStored', [12, 15, 8, 14, 16])
    add_feats(data, feats, 'BitsAllocated', [16, 8])
    add_feats(data, feats, 'HighBit', [11, 14, 7, 13, 15])


def add_feats_pixel_repr_USELESS(data: DataFrame, feats: DataFrame) -> None:
    feats['PixelRepresentation'] = data.PixelRepresentation


def add_feats_samples_per_pixel_USELESS(data: DataFrame, feats: DataFrame) -> None:
    feats['SamplesPerPixel'] = data.SamplesPerPixel


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
    # There are only two genders, so we only need one
    add_feats(data=data, feats=feats, feat="PatientSex", vals=['M'])


def add_feats_photometric(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "PhotometricInterpretation", ['MONOCHROME1', 'MONOCHROME2'])


def add_feats_modality(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "Modality", ['DX', 'CR'])


def add_feats_charset(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "SpecificCharacterSet", ['ISO_IR 100', 'ISO_IR 192'])


def add_feats_deidentification_method(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "DeidentificationMethod", vals=[
        "CTP Default:  based on DICOM PS3.15 AnnexE. Details in 0012,0064",
        "CTP Default",
        "RSNA Covid-19 Dataset Default",
    ])


def add_feats_media_storage(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, "MediaStorageSOPClassUID", vals=[
        "UI: Digital X-Ray Image Storage - For Presentation",
        # This second feat is useless since it's always one or the other
        # "UI: Computed Radiography Image Storage",
    ])


def add_feats_transfer_syntax(data: DataFrame, feats: DataFrame) -> None:
    add_feats(data, feats, 'TransferSyntaxUID', vals=[
        "UI: Explicit VR Little Endian",
        # This second feature is useless since it's always one or the other
        # "UI: JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])",
    ])


def add_tags(data: DataFrame, feats: DataFrame) -> None:
    for col in data.columns:
        if col.startswith('TAG_'):
            feats[col] = data[col]


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
        # add_feats_rows_cols, TODO add back
        # add_feats_bits, TODO add back
        # add_feats_pixel_repr,  <- they are all 0
        # add_feats_pixel_spacing, TODO add back
        # add_feats_samples_per_pixel,  <- they are all 1
        # add_tags, TODO add back
        # add_feats_deidentification_method, TODO add back
        # add_feats_media_storage, TODO add back
        # add_feats_transfer_syntax, TODO add back
        # add_feats_row_col_pixel_spacing, TODO add back
    ]
    for add in to_add:
        add(data=data, feats=feats)

    # Clean up and return
    feats = feats.set_index(["study_id", "image_id"])
    # All boolean feats should be ints
    for col in feats:
        if feats[col].dtype.name == 'bool':
            feats[col] = feats[col].astype(int)

    feats.to_csv(const.subdir_data_csv(path=True) / f'metadata_feats_{sname}.csv')


def verify_feats(feats):
    for feat in feats:
        print(feat, sum(feats[feat]))
