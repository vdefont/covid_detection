"""
When using vastai, first:
- apt-get install gcc
- pip install icevision[all]
"""

from icevision.all import *
import pickle
from typing import Dict, List, Optional

import const


# DATA #

def make_records(box_dir: str, sname: str):
    parser = parsers.VOCBBoxParser(
        annotations_dir=f"{box_dir}/{sname}/annotations",
        images_dir=f"{box_dir}/{sname}/images",
    )
    r1, r2 = parser.parse()
    records = r1 + r2
    print(f"Num records ({sname}): {len(records)}")
    return records


def maybe_modify_tfm(tfm):
    """
    Change or remove transforms we don't want
    """
    name = type(tfm).__name__
    if name == 'ShiftScaleRotate':
        tfm.rotate_limit = (-3, 3)
    elif name in ['RGBShift', 'Blur']:
        tfm = None
    #     elif name == 'OneOrOther':
    #         # This is some aggressive cropping. Keep it for now!
    #         tfm = None
    return tfm


def modify_tfms(tfms):
    """
    Change or remove transforms we don't want
    """
    tfms = [maybe_modify_tfm(tfm=t) for t in tfms]
    return [t for t in tfms if t is not None]


def get_ds_train(box_dir, image_size):
    box_dir = const.subdir_data_detect() + box_dir
    train_tfms = tfms.A.Adapter([
        *modify_tfms(tfms.A.aug_tfms(size=image_size, presize=int(image_size*1.5))),
        tfms.A.Normalize()
    ])
    records = make_records(box_dir=box_dir, sname='train')
    return Dataset(records, train_tfms)


def get_ds_valid(box_dir, image_size, sname='valid'):
    box_dir = const.subdir_data_detect() + box_dir
    valid_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(image_size), tfms.A.Normalize()
    ])
    records = make_records(box_dir=box_dir, sname=sname)
    return Dataset(records, valid_tfms)


def get_ds_train_valid(box_dir, image_size):
    train_ds = get_ds_train(box_dir=box_dir, image_size=image_size)
    valid_ds = get_ds_valid(box_dir=box_dir, image_size=image_size)
    return train_ds, valid_ds


def show_tfmd_imgs(ds, num_to_show=20):
    for i in range(num_to_show):
        # Show three randomly sampled transforms for each image
        samples = [ds[i] for _ in range(3)]
        show_samples(samples, ncols=3)
        plt.pause(0.001)


def get_dl_train_valid(train_ds, valid_ds, model_name, batch_size):
    model_type = model_name_to_type(model_name)

    train_dl = model_type.train_dl(train_ds, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=batch_size, num_workers=4, shuffle=False)
    return train_dl, valid_dl


# MODEL #

def _get_model_eff(backbone, image_size):
    return models.ross.efficientdet.model(
        backbone=backbone(pretrained=True),
        num_classes=2,  # opacity, background (unused)
        img_size=image_size,
    )


MODELS = {
    "eff_lite0_256": _get_model_eff(backbone=models.ross.efficientdet.tf_lite0, image_size=256),
    # "eff_d0_256": _get_model_eff(backbone=models.ross.efficientdet.d0, image_size=256),
}


def model_name_to_type(name):
    if name.startswith("eff"):
        return models.ross.efficientdet
    raise Exception("Could not find model type!")


# METRIC #

def _get_metric():
    return COCOMetric(metric_type=COCOMetricType.bbox, iou_thresholds=np.arange(0, 1.01, 0.1))


# LEARN #

def get_learner(train_dl, valid_dl, model_name, load_model: bool = False):
    model_type = model_name_to_type(model_name)
    model = MODELS[model_name]
    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=[_get_metric()])
    learn.unfreeze()
    if load_model:
        load_dir = const.subdir_models_detect(path=True)
        learn.model.load_state_dict(torch.load(load_dir/model_name))
    return learn


def save_learner(learn: fastai.Learner, model_name: str) -> None:
    save_dir = const.subdir_models_detect(path=True)
    torch.save(learn.model.state_dict(), save_dir / model_name)


# INFERENCE #

def _get_preds(model_type, model, infer_ds):
    infer_dl = model_type.infer_dl(infer_ds, batch_size=8, shuffle=False)
    return model_type.predict_from_dl(model=model, infer_dl=infer_dl, keep_images=True, detection_threshold=0.)


def predict_and_save(box_dir: str, image_size: int, model_name: str, model, test_only: bool = False) -> Dict[str, Any]:
    model_type = model_name_to_type(model_name)

    # When predicting train + valid on vastai, we want to append "_preds"
    # But *not* when predicting test set on kaggle!
    model_name_extn = "" if test_only else "_preds"
    save_dir = const.subdir_preds_detect(path=True) / (model_name + model_name_extn)
    if not save_dir.exists():
        save_dir.mkdir()
    snames = ['test'] if test_only else ['train', 'valid']
    ret = {}
    for sname in snames:
        infer_ds = get_ds_valid(box_dir=box_dir, image_size=image_size, sname=sname)
        ids = [r.as_dict()['common']['filepath'].name[:-len('.png')] for r in infer_ds.records]
        cur_preds = _get_preds(model_type=model_type, model=model, infer_ds=infer_ds)
        cur_preds = [pred.pred.as_dict()['detection'] for pred in cur_preds]
        cur_preds = dict(zip(ids, cur_preds))
        with open(save_dir/sname, 'wb') as f:
            pickle.dump(cur_preds, f)
        ret[sname] = cur_preds
    return ret