from fastai.vision.all import *
import albumentations
import pickle
import torch
from torch import Tensor
from typing import Iterable, Optional, List, Dict, Callable, Tuple, Any
from pathlib import Path
import tqdm

import const


# DATA #


def get_y_neg(path: Path) -> str:
    return 'negative' if path.parent.name == 'neg' else 'positive'


class AlbumentationsTransform(Transform):
    split_idx = 0
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(img_size: int):
    # TUNED (cutout, and tried all others)
    cutout_size = int(img_size * 0.3)
    return albumentations.Compose([
        albumentations.Cutout(num_holes=1, max_h_size=cutout_size, max_w_size=cutout_size, p=0.7),
    ])


class FoldSplitter:
    def __init__(self, fold_valid: int) -> None:
        self.fold_dir_valid = f"fold{fold_valid}"

    def __call__(self, paths: Iterable[Path]) -> Tuple[List[int], List[int]]:
        tr, vl = [], []
        for i, p in enumerate(paths):
            is_valid = p.parent.parent.name == self.fold_dir_valid
            if is_valid:
                vl.append(i)
            else:
                tr.append(i)
        return tr, vl


def get_dls(
        image_path: str,
        img_size: int,
        is_neg: bool,
        presize_amt: float = 1,
        fold_valid: Optional[int] = None,
        test_only: bool = False,
) -> DataLoaders:
    """
    NOTE: This does not work on my machine for some reason, but it works on vastai and kaggle (I think)
    """
    if img_size > 224 and presize_amt == 1:
        raise Exception("You probably should presize!")

    data_block = DataBlock(
        get_items=get_image_files,
        get_y=get_y_neg if is_neg else parent_label,
        blocks=(ImageBlock, CategoryBlock),
        splitter=(
            GrandparentSplitter(valid_name=('test' if test_only else 'valid'))
            if fold_valid is None else FoldSplitter(fold_valid=fold_valid)
        ),
        item_tfms = [Resize(int(img_size * presize_amt)), AlbumentationsTransform(get_train_aug(img_size=img_size))],
        batch_tfms=[
            Brightness(max_lighting=0.05, p=0.75),
            Contrast(max_lighting=0.2, p=0.75),
            *aug_transforms(size=img_size, max_rotate=3, max_warp=0, max_lighting=0, max_zoom=1.0),
            Normalize(),
        ]
    )

    return data_block.dataloaders(const.subdir_data_class(path=True) / image_path, bs=32)


# MODEL #


ARCH_FEATS = {
    resnet18: 512,
    resnet34: 512,
    resnet50: 2048,
    resnet101: 2048,
    resnet152: 2048,
}


def _get_model(arch: Callable, is_neg: bool, head_dropout: float = 0.) -> nn.Module:
    body = create_body(arch, cut=-2)
    head = create_head(ARCH_FEATS[arch], 2 if is_neg else 4, ps=head_dropout)
    return nn.Sequential(body, head)


def _get_model_path(model_name: str, is_neg: bool) -> Path:
    base = const.dir_base_models(path=True)
    extn = "neg" if is_neg else "class"
    return base / extn / model_name


def _save_model(model_name: str, is_neg: bool, model: nn.Module) -> None:
    path = _get_model_path(model_name=model_name, is_neg=is_neg)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def _load_model(model_name: str, is_neg: bool, num_folds: Optional[int], fold_i: Optional[int]) -> nn.Module:
    assert not ((num_folds is None) ^ (fold_i is None))
    if num_folds:
        model_name = f"{model_name}_{num_folds}fold"
    path = _get_model_path(model_name=model_name, is_neg=is_neg)
    if fold_i:
        path = path / f"fold{fold_i}"
    with open(path, 'rb') as f:
        return pickle.load(f)


def make_base_models():
    models_class = {
        "resnet18": _get_model(resnet18, is_neg=False, head_dropout=0.),
    }
    for name, model in models_class.items():
        _save_model(model_name=name, is_neg=False, model=model)

    models_neg = {
        "resnet18": _get_model(resnet18, is_neg=True, head_dropout=0.),
    }
    for name, model in models_neg.items():
        _save_model(model_name=name, is_neg=True, model=model)


# LEARN #


def get_learn(
        dls: DataLoaders,
        model_name: str,
        is_neg: bool,
        load_model: bool = False,
        num_folds: Optional[int] = None,
        fold_i: Optional[int] = None,
) -> Learner:
    # Include "name" to load pre-trained weights
    model = _load_model(model_name=model_name, is_neg=is_neg, num_folds=num_folds, fold_i=fold_i)
    learn = Learner(
        dls=dls,
        model=model,
        loss_func=CrossEntropyLossFlat(),
        # metrics=accuracy,
    )
    learn.unfreeze()
    if load_model:
        load_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
        learn.model.load_state_dict(torch.load(str(load_dir/model_name)))
    return learn


def save_learn(learn: Learner, model_name: str, is_neg: bool) -> None:
    save_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
    torch.save(learn.model.state_dict(), save_dir / model_name)


def save_learn_folds(learn_folds: List[Learner], model_name: str, is_neg: bool) -> None:
    save_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
    save_dir = save_dir / f"{model_name}_{len(learn_folds)}fold"
    save_dir.mkdir()
    for fold_i, learn in enumerate(learn_folds):
        torch.save(learn.model.state_dict(), save_dir / f"fold{fold_i}")


# PREDICT


def _get_dl_names(dl):
    return [p.name.replace('.png', '').replace('.jpg', '') for p in dl.items]


def _get_save_dir(name: str, is_neg: bool) -> Path:
    ret = const.subdir_preds_neg(path=True) if is_neg else const.subdir_preds_class(path=True)
    return ret / name


def predict_and_save(learn: Learner, sname: str, model_name: str, n_tta: int, is_neg: bool) -> Tuple[Tensor, Tensor]:
    idx = 0 if sname == 'train' else 1

    if n_tta > 0:
        preds, targs = learn.tta(idx, n=n_tta)
    else:
        preds, targs = learn.get_preds(ds_idx=idx)

    # When predicting train + valid on vastai, we want to append "_preds"
    # But *not* when predicting test set on kaggle!
    model_name_extn = "" if sname == 'test' else "_preds"
    save_dir = _get_save_dir(name=(model_name + model_name_extn), is_neg=is_neg)

    df = pd.DataFrame(preds.numpy())
    dl = learn.dls.train if sname == 'train' else learn.dls.valid
    df.columns = learn.dls.vocab
    df.index = _get_dl_names(dl=dl)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    df.to_csv(str(save_dir/sname) + ".csv")

    return preds, targs


def predict_and_save_folds(
        learn_folds: List[Learner], model_name: str, n_tta: int, is_neg: bool
) -> Tuple[Tensor, Tensor]:
    """
    Assumptions:
    - This is happening on vastai
    - We are only predicting the validation set
    """
    preds_ls, targs_ls, idx_ls = [], [], []
    for learn in learn_folds:
        if n_tta == 0:
            preds, targs = learn.get_preds(ds_idx=1)
        else:
            preds, targs = learn.tta(n=n_tta)
        preds_ls.append(preds)
        targs_ls.append(targs)
        dl = learn.dls.valid
        idx_ls.append(_get_dl_names(dl=dl))

    preds = torch.cat(preds_ls, axis=0)
    targs = torch.cat(targs_ls)
    idx = np.concatenate(idx_ls)

    df = pd.DataFrame(preds.numpy())
    df.index = idx
    save_dir = _get_save_dir(name=f"{model_name}_{len(learn_folds)}fold", is_neg=is_neg)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    df.to_csv(str(save_dir/f'valid}') + ".csv")

    return preds, targs


# ANALYZE ACCURACY #


def preds_loss(preds: Tensor, targs: Tensor) -> float:
    return -preds[range(len(targs)), targs].log().mean().item()


def analyze_accuracy_by_category(dls: DataLoaders, preds: Tensor, targs: Tensor) -> pd.DataFrame:
    df = pd.DataFrame({'pred': preds.argmax(1).numpy(), 'act': targs.numpy()})
    df['correct'] = df.pred == df.act
    df = df.groupby(df.act).correct.agg(['mean', 'count'])
    df['lab'] = dls.vocab
    return df


def confusion_matrix(learn: Learner) -> None:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()


# SiLU: TODO see if this helps #


def use_silu_rec(parent, idx, component):
    if isinstance(component, Iterable):
        for i, c in enumerate(component):
            use_silu_rec(component, i, c)
    elif isinstance(component, torch.nn.ReLU):
        parent[idx] = torch.nn.SiLU()
    elif hasattr(component, 'relu'):
        component.relu = torch.nn.SiLU()

def use_silu(component):
    for i, c in enumerate(component):
        use_silu_rec(component, i, c)