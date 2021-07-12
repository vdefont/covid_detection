"""
When using vastai, first:
- apt-get install gcc
- pip install icevision[all]
"""

from icevision.all import *
import pickle
from typing import Dict, List, Optional, Any, Tuple
from effdet import DetBenchTrain
import torch
from torch import nn
from make_data_detect import Box, Frame, unscale_bbox_ls
import math
import pandas as pd
import tqdm

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


def get_ds_train(box_dir, image_size):
    box_dir = const.subdir_data_detect() + box_dir
    train_tfms = tfms.A.Adapter([
        tfms.A.Resize(image_size, image_size),
        tfms.A.Normalize(),
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
    from icevision.visualize import show_samples
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


def _get_model(name: str):
    if name == "eff_lite0_256":
        return _get_model_eff(backbone=models.ross.efficientdet.tf_lite0, image_size=256)
    if name == "eff_d0_256":
        return _get_model_eff(backbone=models.ross.efficientdet.d0, image_size=256)
    raise Exception(f"No model called {name}")


def make_base_models():
    base_models = {n: _get_model(n) for n in ["eff_lite0_256"]}
    for name, model in base_models.items():
        path = const.dir_base_models(path=True)/'detect'/name
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def _load_model(model_name: str) -> nn.Module:
    def param_groups_fn(model: nn.Module) -> List[List[nn.Parameter]]:
        pass

    # This wack stuff is necessary for pickle.load to work
    DetBenchTrain.param_groups_fn = param_groups_fn

    path = const.dir_base_models(path=True)/'detect'/model_name
    with open(path, 'rb') as f:
        return pickle.load(f)


def model_name_to_type(name):
    if name.startswith("eff"):
        return models.ross.efficientdet
    raise Exception("Could not find model type!")


# METRIC #


def _iou(b1: Box, b2: Box) -> float:
    x_s = max(b1.X, b2.X)
    x_e = min(b1.X + b1.W, b2.X + b2.W)
    y_s = max(b1.Y, b2.Y)
    y_e = min(b1.Y + b1.H, b2.Y + b2.H)

    if x_s >= x_e or y_s >= y_e:
        return 0

    intersection = (x_e - x_s) * (y_e - y_s)
    union = b1.W * b1.H + b2.W * b2.H - intersection
    return intersection / union


def test_iou():
    def _test(b1, b2, expect):
        print(_iou(b1, b2))
        print(_iou(b2, b1))
        print(f"(expected {expect})\n")

    b1 = Box(0,0,2,2)
    b2 = Box(1,1,2,2)
    _test(b1, b2, 1/7)

    b1 = Box(0, 0, 4, 3)
    b2 = Box(2, 1, 3, 5)
    _test(b1, b2, 4/23)

    b1 = Box(0,0,5,2)
    b2 = Box(2,2,5,2)
    _test(b1, b2, 0)

    b1 = Box(0, 0, 2, 5)
    b2 = Box(2, 2, 2, 5)
    _test(b1, b2, 0)


def _precision(actuals_preds_ls: List[Tuple[List[Box], List[Box]]], recall: float) -> float:
    if recall == 0.0:
        return 1

    total_boxes = sum(len(ap[0]) for ap in actuals_preds_ls)
    num_needed = math.ceil(total_boxes * recall)

    # List out the prediction indices with highest-confidence boxes first
    row_col_conf_ls = [
        (row, col, box.C)
        for row, actuals_preds in enumerate(actuals_preds_ls)
        for col, box in enumerate(actuals_preds[1])
    ]
    row_col_conf_ls = sorted(row_col_conf_ls, key=lambda rcc: -rcc[2])

    num_found = 0
    num_guessed = 0
    actuals_ls = [deepcopy(actuals_preds[0]) for actuals_preds in actuals_preds_ls]
    for row, col, _conf in row_col_conf_ls:
        if num_found == num_needed:
            break

        # See if the current prediction matches one of the actual boxes
        actuals = actuals_ls[row]
        pred = actuals_preds_ls[row][1][col]
        found_i = None
        for i, actual in enumerate(actuals):
            if _iou(pred, actual) > 0.5:
                found_i = i
                break

        if found_i is not None:
            actuals.pop(found_i)  # We can't predict this box again once we've already got it
            num_found += 1

        num_guessed += 1

    if num_found < num_needed:
        return 0
    return num_found / num_guessed


def mean_average_precision(actuals_preds_ls: List[Tuple[List[Box], List[Box]]], verbose=False):
    precisions = [
        _precision(actuals_preds_ls=actuals_preds_ls, recall=r)
        for r in tqdm.tqdm(np.arange(0, 1.01, 0.1), desc="Calculating mAP")
    ]
    if verbose:
        print(f"Precisions: {precisions}")
    return np.mean(precisions)


def test_map():
    def _test(actuals_preds_ls):
        for recall in np.arange(0, 1.01, 0.1):
            print(f"\nRecall: {recall}")
            precision = _precision(actuals_preds_ls, recall)
            print(f"Precision: {precision}")

        map = mean_average_precision(actuals_preds_ls)
        print(f"\nMAP: {map}")

    def b1(c=0): return Box(X=0, Y=0, W=2, H=3, C=-c)
    def b2(c=0): return Box(X=4, Y=3, W=2, H=2, C=-c)
    def b3(c=0): return Box(X=0, Y=3, W=2, H=2, C=-c)
    actuals_preds_ls = [
        ([b1(), b2()], [b1(1), b3(3), b1(5), b3(7), b1(9), b2(11), b2(12), b2(13), b2(14)]),
        ([b3()], [b1(2), b3(4), b1(6), b3(8), b1(10)]),
    ]
    _test(actuals_preds_ls) # Expect 0.599

    # This time, we don't ever get b2
    actuals_preds_ls = [
        ([b1(), b2()], [b1(1), b3(3), b1(5), b3(7), b1(9)]),
        ([b3()], [b1(2), b3(4), b1(6), b3(8), b1(10)]),
    ]
    _test(actuals_preds_ls) # Expect 0.5


def _get_metric_DEPRECATED():
    return COCOMetric(metric_type=COCOMetricType.bbox, iou_thresholds=np.arange(0, 1.01, 0.1))


class MapMetric(Metric):
    def __init__(self):
        self._records, self._preds = [], []
        self.i = 0

    def _reset(self):
        self._records.clear()
        self._preds.clear()

    def accumulate(self, preds):
        for pred in preds:
            self._records.append(pred.ground_truth)
            self._preds.append(pred.pred)

    def _make_box(self, bbox, score=None) -> Box:
        return Box(
            X=bbox.xmin,
            Y=bbox.ymin,
            W=bbox.xmax - bbox.xmin,
            H=bbox.ymax - bbox.ymin,
            C=score,
        )

    def finalize(self) -> Dict[str, float]:
        actuals_preds_ls = []
        for i, (actuals_raw, preds_raw) in enumerate(zip(self._records, self._preds)):
            actuals = [self._make_box(box) for box in actuals_raw.detection.bboxes]
            preds = [
                self._make_box(box, score)
                for box, score in zip(preds_raw.detection.bboxes, preds_raw.detection.scores)
            ]
            actuals_preds_ls.append((actuals, preds))

        map = mean_average_precision(actuals_preds_ls=actuals_preds_ls)
        self._reset()
        return {"mAP": map}


# LEARN #

def get_learner(train_dl, valid_dl, model_name, load_model: bool = False):
    model_type = model_name_to_type(model_name)
    if load_model:
        model = _load_model(model_name=model_name)
    else:
        model = _get_model(name=model_name)

    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=[MapMetric()])

    # For some reason unfreeze() does not work at test time, so let's
    # only do it when training
    if not load_model:
        learn.unfreeze()

    if load_model:
        load_dir = const.subdir_models_detect(path=True)
        learn.model.load_state_dict(torch.load(load_dir/model_name))

    config = dict(learn.model.config)
    config['act_type'] = 'silu'
    model.config = config

    return learn


def save_learner(learn: Any, model_name: str) -> None:
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


# FETCH BOXES FROM OUTPUT FILE #


def unscale_boxes(new_frame: Frame, boxes: Dict[str, List[Box]], sname: str) -> Dict[str, List[Box]]:
    """
    Scale boxes up to original size
    new_frame: The transformed frame that we used for predicting
    """
    id_orig_size = "id_orig_size_test.csv" if sname == "test" else "id_orig_size.csv"
    size_data = pd.read_csv(const.subdir_data_csv(path=True) / id_orig_size)
    id_to_frame = {id: Frame(W=w, H=h) for id, w, h in zip(size_data.id, size_data.width, size_data.height)}
    return {
        id: unscale_bbox_ls(frame=new_frame, new_frame=id_to_frame[id], boxes=bs)
        for id, bs in tqdm.tqdm(boxes.items(), desc="Unscaling boxes")
    }

def _non_max_suppression(boxes: List[Box]) -> List[Box]:
    """
    I observe an increase 0.14332 -> 0.14349
    - Almost nothing, but...
    """
    boxes = sorted(boxes, key=lambda box: -box.C)
    ret = [boxes[0]]
    for box in boxes[1:]:
        max_iou = max(_iou(box, ret_box) for ret_box in ret)
        # mult = 5 ** -max_iou
        # ret.append(Box(X=box.X, Y=box.Y, W=box.W, H=box.H, C=box.C * mult))
        if max_iou <= 0.3:
            ret.append(box)
    return ret


def get_box_preds(
        preds_path_detect: Path, new_frame: Frame, sname: str = 'test', nms: bool = False
) -> Dict[str, List[Box]]:
    """
    Returns: id -> boxes
    """

    with open(preds_path_detect/sname, 'rb') as f:
        preds = pickle.load(f)

    boxes_ls = [p['bboxes'] for p in preds.values()]
    scores_ls = [p['scores'] for p in preds.values()]
    ret = {
        id: [Box(X=box.xmin, Y=box.ymin, W=box.width, H=box.height, C=score) for box, score in zip(boxes, scores)]
        for id, boxes, scores in zip(preds, boxes_ls, scores_ls)
    }
    ret = unscale_boxes(new_frame=new_frame, boxes=ret, sname=sname)

    # Optionall apply non-max suppression
    if nms:
        for id, boxes in tqdm.tqdm(ret.items(), desc="Non-max suppression"):
            ret[id] = _non_max_suppression(boxes=boxes)

    return ret

def test_non_max_suppression() -> None:
    boxes = [
        Box(0,0,1,10,1),
        Box(0,0,1,7,0.9), # Discard
        Box(0,0,1,5,0.8),
        Box(0,0,1,4,0.7), # Discard
        Box(0,0,1,2,0.6),
        Box(0,0,1,1.5,0.5), # Discard
    ]
    for box in _non_max_suppression(boxes):
        print(box)


def test_preds_map(sname: str = 'valid', nms: bool = False) -> None:
    with open(const.subdir_data_detect(path=True)/'png224_boxes'/sname, "rb") as f:
        id_to_actuals = pickle.load(f)
    id_to_actuals = unscale_boxes(new_frame=Frame(224, 224), boxes=id_to_actuals, sname=sname)
    id_to_preds = get_box_preds(
        const.subdir_preds_detect(path=True) / 'eff_lite0_256', new_frame=Frame(W=256, H=256), sname=sname, nms=nms
    )

    assert id_to_actuals.keys() == id_to_preds.keys()

    actuals_preds_ls = []
    for id, actuals in id_to_actuals.items():
        actuals_preds_ls.append((actuals, id_to_preds[id]))

    print(mean_average_precision(actuals_preds_ls=actuals_preds_ls, verbose=True))
    # Currently, precisions =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]