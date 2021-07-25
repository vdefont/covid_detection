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
import re
import functools

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


def get_ds_train(box_dir, image_size, fold_valid):
    box_dir = const.subdir_data_detect() + box_dir
    train_tfms = tfms.A.Adapter([
        tfms.A.Resize(image_size, image_size),
        tfms.A.Normalize(),
    ])
    records = []
    for subdir in Path(box_dir).glob('*'):
        if (not subdir.is_dir()) or subdir.name == f"fold{fold_valid}":
            continue
        records += make_records(box_dir=box_dir, sname=subdir.name)
    return Dataset(records, train_tfms)


def get_ds_valid(box_dir, image_size, sname='valid'):
    box_dir = const.subdir_data_detect() + box_dir
    valid_tfms = tfms.A.Adapter([
        *tfms.A.resize_and_pad(image_size), tfms.A.Normalize()
    ])
    records = make_records(box_dir=box_dir, sname=sname)
    return Dataset(records, valid_tfms)


def get_ds_train_valid(box_dir: str, image_size: int, fold_valid: int):
    train_ds = get_ds_train(box_dir=box_dir, image_size=image_size, fold_valid=fold_valid)
    valid_ds = get_ds_valid(box_dir=box_dir, image_size=image_size, sname=f"fold{fold_valid}")
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
        backbone=backbone,
        num_classes=2,  # opacity, background (unused)
        img_size=image_size,
    )


def _get_model_yolo(backbone, image_size):
    return models.ultralytics.yolov5.model(
        backbone=backbone,
        num_classes=2,  # opacity, background (unused)
        img_size=image_size,
    )


def _get_model_resnet(backbone, image_size):
    return models.torchvision.retinanet.model(
        backbone=backbone,
        num_classes=2,  # opacity, background (unused)
        min_size=image_size,
        max_size=image_size,
        score_thresh=0.001,
    )


ROSS_SHORT_TO_LONG = {}
for i in range(5):
    ROSS_SHORT_TO_LONG[f"eff_lite{i}"] = f"tf_efficientdet_lite{i}"
for i in range(8):
    ROSS_SHORT_TO_LONG[f"eff_tf{i}"] = f"tf_efficientdet_d{i}"
    ROSS_SHORT_TO_LONG[f"eff_d{i}"] = f"efficientdet_d{i}"
for i in range(6):
    ROSS_SHORT_TO_LONG[f"eff_ap{i}"] = f"tf_efficientdet_d{i}_ap"


EFFDET_IMG_SIZES = {
    0: 512, 1: 640, 2: 768, 3: 896, 4: 1024, 5: 1280, 6: 1280, 7: 1536
}


def _recommended_size(model_name: str) -> int:
    idx = int(re.search(r'(\d)', model_name)[1])
    return EFFDET_IMG_SIZES[idx]


def _get_model(name: str, pretrained: bool = True):
    """
    Name can look like "eff_lite0" or "eff_lite0_256"
    ...or "yolo5s_256"
    ...or "retinanet
    """
    image_size = None

    # Handle "lite0_256" case
    re_suffix = re.search(r'_(\d*)$', name)
    if re_suffix is not None:
        image_size = int(re_suffix[1])
        name = name[:-len(re_suffix[0])]

    if name.startswith("eff_"):
        full_name = ROSS_SHORT_TO_LONG[name]
        backbone = models.ross.efficientdet.utils.EfficientDetBackboneConfig(full_name)(pretrained=pretrained)
        image_size = image_size or _recommended_size(model_name=full_name)
        return _get_model_eff(backbone=backbone, image_size=image_size)

    if name.startswith("yolo"):
        # Options:
        #  yolov5s  yolov5m  yolov5l
        #  yolov5s6 yolov5m6 yolov5l6
        backbone = models.ultralytics.yolov5.utils.YoloV5BackboneConfig(model_name=name)(pretrained=pretrained)
        assert image_size is not None
        return _get_model_yolo(backbone=backbone, image_size=image_size)

    if name.startswith("resnet"):
        """
        Options:
        resnet18_fpn - 18, 34, 50, 101, 152
        resnext50_32x4d_fpn
        resnext101_32x8d_fpn
        wide_resnet50_2_fpn
        wide_resnet101_2_fpn
        """
        backbone_fn = functools.partial(backbones._resnet_fpn, name=name, pretrained=pretrained)
        backbone = models.torchvision.retinanet.backbones.resnet_fpn.RetinanetTorchvisionBackboneConfig(backbone_fn)
        assert image_size is not None


    raise Exception(f"Name {name} not recognized!")


def make_base_models(base_models: Iterable[str]):
    # Ex: base_models=["eff_lite0_256"]
    base_models = {n: _get_model(n) for n in base_models}
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

    # Sometimes the model name is formatted like resnet18_s224
    # We want to strip that end part
    match = re.match(r'(.*)_s\d*$', model_name)
    if match:
        model_name = match[1]

    path = const.dir_base_models(path=True)/'detect'/model_name
    with open(path, 'rb') as f:
        return pickle.load(f)


def model_name_to_type(name):
    if name.startswith("eff"):
        return models.ross.efficientdet
    if name.startswith("yolo"):
        return models.ultralytics.yolov5
    if name.startswith("resnet"):
        return models.torchvision.retinanet
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

def get_learner(
        train_dl, valid_dl, model_name, load_model_fold: Optional[int] = None, no_init_load: bool = False
):
    model_type = model_name_to_type(model_name)
    if load_model_fold is not None and not no_init_load:
        model = _load_model(model_name=model_name)
    else:
        model = _get_model(name=model_name)

    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=[MapMetric()])

    # For some reason unfreeze() does not work at test time, so let's
    # only do it when training
    if load_model_fold is None:
        learn.unfreeze()
    else:
        load_dir = const.subdir_models_detect(path=True)
        load_path = load_dir/f"{model_name}"/f"fold{load_model_fold}"
        print(f"LOAD PATH: {load_path}")
        map_location = None if torch.cuda.is_available() else torch.device("cpu")
        learn.model.load_state_dict(torch.load(load_path, map_location=map_location))

    if model_name.startswith("eff"):
        config = dict(learn.model.config)
        config['act_type'] = 'silu'
        model.config = config

    return learn


def save_learn(learn: Any, model_name: str) -> None:
    save_dir = const.subdir_models_detect(path=True)
    torch.save(learn.model.state_dict(), save_dir / model_name)


def save_learn_folds(learn_folds: List[Any], model_name: str) -> None:
    save_dir = const.subdir_models_detect(path=True)
    save_dir = save_dir / f"{model_name}_{len(learn_folds)}fold"
    save_dir.mkdir()
    for fold_i, learn in enumerate(learn_folds):
        torch.save(learn.model.state_dict(), save_dir / f'fold{fold_i}')


# INFERENCE #

def _get_preds(model_type, model, infer_ds):
    # batch size = 1 to not overload kaggle's gpu (?)
    infer_dl = model_type.infer_dl(infer_ds, batch_size=1, shuffle=False)
    return model_type.predict_from_dl(model=model, infer_dl=infer_dl, keep_images=False, detection_threshold=0.)


def predict_and_save(box_dir: str, image_size: int, model_name: str, model: Any, fold: int, dummy: bool = False, is_test: bool = True) -> Any:
    """
    Assumes that we are predicting the test set

    fold: Use the model for the given fold
    """
    model_type = model_name_to_type(model_name)

    save_dir = const.subdir_preds_detect(path=True) / model_name
    if not save_dir.exists():
        save_dir.mkdir()

    infer_ds = get_ds_valid(box_dir=box_dir, image_size=image_size, sname="test" if is_test else f"fold{fold}")
    ids = [r.as_dict()['common']['filepath'].name[:-len('.png')] for r in infer_ds.records]
    if dummy:
        cur_preds = "dummy"
    else:
        cur_preds = _get_preds(model_type=model_type, model=model, infer_ds=infer_ds)
        cur_preds = [pred.pred.as_dict()['detection'] for pred in cur_preds]
        cur_preds = dict(zip(ids, cur_preds))
    with open(save_dir/f"fold{fold}", 'wb') as f:
        pickle.dump(cur_preds, f)

    return cur_preds


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


def _non_max_suppression(boxes: List[Box], thresh: float) -> List[Box]:
    """
    Thresh: if it exceeds this IOU, it is cut
    """
    boxes = sorted(boxes, key=lambda box: -box.C)
    ret = [boxes[0]]
    for box in boxes[1:]:
        max_iou = max(_iou(box, ret_box) for ret_box in ret)
        # mult = 5 ** -max_iou
        # ret.append(Box(X=box.X, Y=box.Y, W=box.W, H=box.H, C=box.C * mult))
        if max_iou <= thresh:
            ret.append(box)
    return ret


def get_box_preds(
        preds_path_detect: Path, new_frame: Frame, sname: str = 'test', fold: Optional[int] = None
) -> Dict[str, List[Box]]:
    """
    Returns: id -> boxes
    """

    preds_name = sname if fold is None else f"fold{fold}"
    with open(preds_path_detect/preds_name, 'rb') as f:
        preds = pickle.load(f)

    boxes_ls = [p['bboxes'] for p in preds.values()]
    scores_ls = [p['scores'] for p in preds.values()]
    ret = {
        id: [Box(X=box.xmin, Y=box.ymin, W=box.width, H=box.height, C=score) for box, score in zip(boxes, scores)]
        for id, boxes, scores in zip(preds, boxes_ls, scores_ls)
    }
    ret = unscale_boxes(new_frame=new_frame, boxes=ret, sname=sname)

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


def test_preds_map(sname: str = 'valid') -> None:
    with open(const.subdir_data_detect(path=True)/'png224_boxes'/sname, "rb") as f:
        id_to_actuals = pickle.load(f)
    id_to_actuals = unscale_boxes(new_frame=Frame(256, 256), boxes=id_to_actuals, sname=sname)
    id_to_preds = get_box_preds(
        const.subdir_preds_detect(path=True) / 'eff_lite0_256', new_frame=Frame(W=256, H=256), sname=sname
    )

    assert id_to_actuals.keys() == id_to_preds.keys()

    actuals_preds_ls = []
    for id, actuals in id_to_actuals.items():
        actuals_preds_ls.append((actuals, id_to_preds[id]))

    print(mean_average_precision(actuals_preds_ls=actuals_preds_ls, verbose=True))
    # Currently, precisions =  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]