from fastai.vision.all import *
import pickle
import torch
from torch import Tensor
from typing import Iterable, Optional, List, Dict, Callable, Tuple, Any
from pathlib import Path

import const


# DATA #


def get_y_neg(path: Path) -> str:
    return 'negative' if path.parent.name == 'neg' else 'positive'


def get_dls(image_path: str, is_neg: bool, test_only: bool = False) -> DataLoaders:
    data_block = DataBlock(
        get_items=get_image_files,
        get_y=get_y_neg if is_neg else parent_label,
        blocks=(ImageBlock, CategoryBlock),
        splitter=GrandparentSplitter(valid_name=('test' if test_only else 'valid')),
        batch_tfms=[
            *aug_transforms(max_rotate=3, max_warp=0.),
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


MODELS: Dict[str, nn.Module] = {
    "resnet18": _get_model(resnet18, is_neg=False, head_dropout=0.),
}


MODELS_NEG: Dict[str, nn.Module] = {
    "resnet18": _get_model(resnet18, is_neg=True, head_dropout=0.),
}


# LEARN #


def get_learn(
        dls: DataLoaders, model_name: str, is_neg: bool, load_model: bool = False
) -> Learner:
    # Include "name" to load pre-trained weights
    model_dict = MODELS_NEG if is_neg else MODELS
    model = model_dict[model_name]
    learn = Learner(
        dls=dls,
        model=model,
        loss_func=CrossEntropyLossFlat(),
        metrics=accuracy,
    )
    learn.unfreeze()
    if load_model:
        load_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
        learn.model.load_state_dict(torch.load(str(load_dir/model_name)))
    return learn


def save_learn(learn: Learner, model_name: str, is_neg: bool) -> None:
    save_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
    torch.save(learn.model.state_dict(), save_dir / model_name)


# PREDICT


def _get_dl_names(dl):
    return [p.name.replace('.png', '') for p in dl.items]


def _get_save_dir(name: str, is_neg: bool) -> Path:
    ret = const.subdir_preds_neg(path=True) if is_neg else const.subdir_preds_class(path=True)
    return ret / name


def predict_and_save(learn: Learner, sname: str, model_name: str, is_neg: bool) -> Tuple[Tensor, Tensor]:
    idx = 0 if sname == 'train' else 1
    preds, targs = learn.tta(idx)

    save_dir = _get_save_dir(name=model_name + "_preds", is_neg=is_neg)

    df = pd.DataFrame(preds.numpy())
    dl = learn.dls.train if sname == 'train' else learn.dls.valid
    df.columns = learn.dls.vocab
    df.index = _get_dl_names(dl=dl)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir()
    df.to_csv(str(save_dir/sname) + ".csv")

    return preds, targs


# ANALYZE ACCURACY #


def analyze_accuracy_by_category(dls: DataLoaders, preds: Tensor, targs: Tensor) -> pd.DataFrame:
    df = pd.DataFrame({'pred': preds.argmax(1).numpy(), 'act': targs.numpy()})
    df['correct'] = df.pred == df.act
    df = df.groupby(df.act).correct.agg(['mean', 'count'])
    df['lab'] = dls.vocab
    return df


def confusion_matrix(learn: Learner) -> None:
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()


# RRELU: Doesn't really help #


def use_rrelu_rec(parent, idx, component):
    if isinstance(component, Iterable):
        for i, c in enumerate(component):
            use_rrelu_rec(component, i, c)
    elif isinstance(component, torch.nn.ReLU):
        parent[idx] = torch.nn.RReLU()
    else:
        try:
            component.relu = torch.nn.RReLU()
        except:
            pass

def use_rrelu(component):
    for i, c in enumerate(component):
        use_rrelu_rec(component, i, c)