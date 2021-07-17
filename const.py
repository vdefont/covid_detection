from pathlib import Path
from typing import Union
from enum import Enum


DIR_BASE_MODELS = "./base_models/"
DIR_ORIGINAL_DATA = "./data_original/"
DIR_WORKING = "./"
# Subdirs of working dir
SUBDIR_DATA_IMAGE = "data_image/"
SUBDIR_DATA_CSV = "data_csv/"
SUBDIR_DATA_CLASS = "data_class/"
SUBDIR_DATA_DETECT = "data_detect/"
SUBDIR_MODELS_CLASS = "models_class/"
SUBDIR_PREDS_CLASS = "preds_class/"
SUBDIR_MODELS_NEG = "models_neg/"
SUBDIR_PREDS_NEG = "preds_neg/"
SUBDIR_MODELS_XGB = "models_xgb/"
SUBDIR_MODELS_DETECT = "models_detect/"
SUBDIR_PREDS_DETECT = "preds_detect/"
SUBDIR_PREDS_FINAL = "preds_final/"
SUBDIR_MODELS_KAGGLE = "models_kaggle/"


class Vocab(Enum):
    NEG = "neg"
    TYP = "typ"
    IND = "ind"
    ATYP = "atyp"


VOCAB_SHORT = [Vocab.NEG.value, Vocab.TYP.value, Vocab.IND.value, Vocab.ATYP.value]
VOCAB_SHORT_TO_LONG = {
    Vocab.NEG.value: 'negative',
    Vocab.TYP.value: 'typical',
    Vocab.IND.value: 'indeterminate',
    Vocab.ATYP.value: 'atypical',
}
VOCAB_LONG = [VOCAB_SHORT_TO_LONG[v] for v in VOCAB_SHORT]
VOCAB_FULL = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]

SNAMES = ["train", "valid", "test"]


def maybe_path(dir_func):
    def f(path: bool = False) -> Union[Path, str]:
        ret = dir_func()
        return Path(ret) if path else ret
    return f


@maybe_path
def dir_base_models():
    return DIR_BASE_MODELS


@maybe_path
def dir_original_data():
    return DIR_ORIGINAL_DATA


@maybe_path
def dir_working():
    return DIR_WORKING


@maybe_path
def subdir_data_image():
    return DIR_WORKING + SUBDIR_DATA_IMAGE


@maybe_path
def subdir_data_csv():
    return DIR_WORKING + SUBDIR_DATA_CSV


@maybe_path
def subdir_data_class():
    return DIR_WORKING + SUBDIR_DATA_CLASS


@maybe_path
def subdir_data_detect():
    return DIR_WORKING + SUBDIR_DATA_DETECT


@maybe_path
def subdir_models_class():
    return DIR_WORKING + SUBDIR_MODELS_CLASS


@maybe_path
def subdir_preds_class():
    return DIR_WORKING + SUBDIR_PREDS_CLASS


@maybe_path
def subdir_models_neg():
    return DIR_WORKING + SUBDIR_MODELS_NEG


@maybe_path
def subdir_preds_neg():
    return DIR_WORKING + SUBDIR_PREDS_NEG


@maybe_path
def subdir_models_xgb():
    return DIR_WORKING + SUBDIR_MODELS_XGB


@maybe_path
def subdir_models_detect():
    return DIR_WORKING + SUBDIR_MODELS_DETECT


@maybe_path
def subdir_preds_detect():
    return DIR_WORKING + SUBDIR_PREDS_DETECT


@maybe_path
def subdir_preds_final():
    return DIR_WORKING + SUBDIR_PREDS_FINAL


@maybe_path
def subdir_models_kaggle():
    return DIR_WORKING + SUBDIR_MODELS_KAGGLE