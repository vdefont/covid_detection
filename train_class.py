from fastai.vision.all import *
import albumentations
import pickle
import torch
from torch import Tensor
from typing import Iterable, Optional, List, Dict, Callable, Tuple, Any, Union
from pathlib import Path
import tqdm
from PIL import Image
import timm
import re
import functools

import const


# DATA #


def get_y_neg(neg_cls: const.Vocab, path: Path) -> str:
    if neg_cls == const.Vocab.NEG:
        return const.VocabNeg.NEG.value if path.parent.name == const.Vocab.NEG.value else const.VocabNeg.POS.value
    cls_str = neg_cls.value
    return cls_str if path.parent.name == cls_str else f"not_{cls_str}"


class AlbumentationsTransform(Transform):
    split_idx = 0
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def get_train_aug(orig_img_size: int):
    # TUNED (cutout, and tried all others)
    cutout_size = int(orig_img_size * 0.3)
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
        neg_cls: const.Vocab = const.Vocab.NEG,
        presize_amt: float = 1,
        fold_valid: Optional[int] = None,
        test_only: bool = False,
        batch_size: int = 32,
        flip_p: float = 0.5,
        normalize_stats: Optional[Any] = None,
        get_items=get_image_files,
) -> DataLoaders:
    """
    NOTE: This does not work on my machine for some reason, but it works on vastai and kaggle (I think)
    """
    orig_img_size = int(re.match(r'[^\d]*(\d*)', image_path).group(1))

    data_block = DataBlock(
        get_items=get_items,
        get_y=functools.partial(get_y_neg, neg_cls) if is_neg else parent_label,
        blocks=(ImageBlock, CategoryBlock),
        splitter=(
            GrandparentSplitter(valid_name=('test' if test_only else 'valid'))
            if fold_valid is None else FoldSplitter(fold_valid=fold_valid)
        ),
        item_tfms=[
            Resize(int(img_size * presize_amt)),
            AlbumentationsTransform(get_train_aug(orig_img_size=orig_img_size))
        ],
        batch_tfms=[
            Brightness(max_lighting=0.05, p=0.75),
            Contrast(max_lighting=0.2, p=0.75),
            *aug_transforms(size=img_size, max_rotate=3, max_warp=0, max_lighting=0, max_zoom=1.0),
            Normalize.from_stats(*normalize_stats) if normalize_stats is not None else Normalize(),
        ]
    )

    return data_block.dataloaders(const.subdir_data_class(path=True) / image_path, bs=batch_size)


# MODEL #


ARCH_CUT = {
    resnet18: -2,
    resnet34: -2,
    resnet50: -2,
    resnet101: -2,
    resnet152: -2,
    densenet121: -1,
    densenet161: -1,
    densenet169: -1,
    densenet201: -1,
}


TIMM_SHORT_TO_LONG = {}
for i in range(5):
    TIMM_SHORT_TO_LONG[f"efflite{i}"] = f"efficientnet_lite{i}"
for i in range(9):
    TIMM_SHORT_TO_LONG[f"effnet{i}"] = f"efficientnet_b{i}"


def _create_timm_body(arch: str, pretrained: bool) -> nn.Module:
    "Creates a body from any model in the `timm` library."
    model = timm.create_model(TIMM_SHORT_TO_LONG.get(arch, arch), pretrained=pretrained, num_classes=0, global_pool='')
    ll = list(enumerate(model.children()))
    cut = next(i for i,o in reversed(ll) if has_pool_type(o))
    return nn.Sequential(*list(model.children())[:cut])


def _get_body(arch: Union[Callable, str], pretrained: bool = True) -> nn.Module:
    if isinstance(arch, Callable):
        return create_body(arch, cut=ARCH_CUT[arch], pretrained=pretrained)
    assert isinstance(arch, str)
    return _create_timm_body(arch=arch, pretrained=pretrained)


def _get_model(body: nn.Module, is_neg: bool, head_dropout: float = 0.) -> nn.Module:
    head = create_head(num_features_model(body), 2 if is_neg else 4, ps=head_dropout)
    return nn.Sequential(body, head)


def get_model(arch: Union[Callable, str], is_neg: bool, head_dropout: float = 0., pretrained: bool = True) -> nn.Module:
    body = _get_body(arch=arch, pretrained=pretrained)
    return _get_model(body=body, is_neg=is_neg, head_dropout=head_dropout)


def _get_model_path(model_name: str) -> Path:
    return const.dir_base_models(path=True) / 'class' / model_name


def _save_model(model_name: str, model: nn.Module) -> None:
    path = _get_model_path(model_name=model_name)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def make_base_models(model_names: Iterable[Union[Callable, str]]):
    """
    Saves the model bodies
    """
    for name in model_names:
        model = _get_body(arch=name)
        name_str = name if isinstance(name, str) else name.__name__
        _save_model(model_name=name_str, model=model)


def _load_model(model_name: str, is_neg: bool) -> nn.Module:
    """
    Loads the model, after attaching the appropriate head
    """

    # Sometimes the model name is formatted like resnet18_s224
    # We want to strip that end part
    match = re.match(r'(.*)_s\d*$', model_name)
    if match:
        model_name = match[1]

    path = _get_model_path(model_name=model_name)
    with open(path, 'rb') as f:
        body = pickle.load(f)
    return _get_model(body=body, is_neg=is_neg)


def model_size(m: Union[nn.Module, Callable, str]) -> None:
    if not isinstance(m, nn.Module):
        m = timm.create_model(m, pretrained=False, num_classes=0, global_pool='')

    # Prints number of params, in millions
    ret = 0
    for p in m.parameters():
        ret += p.numel()
    ret /= 1_000_000
    print(f"{ret:.1f}m parameters")


# METRIC #


def single_map(probs: np.ndarray, targs: np.ndarray):
    """
    targs: 0 or 1
    """
    prob_targs = sorted(zip(probs, targs), key=lambda pt: -pt[0])
    targs = np.array([pt[1] for pt in prob_targs]).astype(int)

    map_acc = 1.
    incr = targs.sum() / 10
    target = incr
    correct_so_far = 0
    for i, t in enumerate(targs):
        correct_so_far += t
        while correct_so_far >= target:
            map_acc += correct_so_far / (i + 1)
            target += incr
    return map_acc / 11


def test_single_map():
    probs = [1,.8,.6,.4,.2,.9,.7,.5,.3,.1]
    targs = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    map = single_map(probs, targs)
    print(f"MAP: {map}")
    expected = np.mean([1,1,2/3,2/3,3/4,3/4,4/6,5/8,5/8,6/10,6/10])
    print(f"Expected: {expected}")


def preds_map(preds, targs) -> float:
    if not isinstance(preds, np.ndarray):
        preds, targs = np.array(preds.cpu()), np.array(targs.cpu())
    maps = []
    for i in range(preds.shape[1]):
        probs_i = preds[:, i]
        targs_i = np.array(targs) == i
        if np.sum(targs_i) == 0:
            print(f"No {i}.", end="")
            continue
        maps.append(single_map(probs_i, targs_i))
    return np.mean(maps) * 2/3


def test_preds_map():
    preds = np.array([
        [0,1.],
        [1,.9],
        [0,.8],
        [0,.7],
        [1,.6],
        [0,.5],
        [1,.4],
        [0,.3],
        [1,.2],
        [0,.1],
    ])
    targs = [1,0,1,1,0,1,0,1,0,1]
    pm = preds_map(preds, targs)
    print(f"Actual: {pm}")
    print(f"Expected: {(0.7227272727272727 + 1) / 2 * 2 / 3}")

# LEARN #


def get_learn(
        dls: DataLoaders,
        model_name: str,
        is_neg: bool,
        load_model_fold: Optional[int] = None,
        loss_func: Optional[Any] = None,
        cbs: Optional[Any] = None,
) -> Learner:
    # Include "name" to load pre-trained weights
    model = _load_model(model_name=model_name, is_neg=is_neg)
    learn = Learner(
        dls=dls,
        model=model,
        loss_func=CrossEntropyLossFlat() if loss_func is None else loss_func,
        # metrics=[AccumMetric(preds_map)],
        cbs=cbs,
    )
    learn.unfreeze()
    if load_model_fold is not None:
        load_dir = const.subdir_models_neg(path=True) if is_neg else const.subdir_models_class(path=True)
        map_location = None if torch.cuda.is_available() else torch.device('cpu')
        learn.model.load_state_dict(torch.load(
            str(load_dir/model_name/f"fold{load_model_fold}"),
            map_location=map_location,
        ))
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


def predict_and_save(
        learn: Union[Learner, List[Learner]],
        sname: str,
        model_name: str,
        n_tta: int,
        is_neg: bool,
        dls_tta: Optional[Any] = None,
) -> Tuple[Tensor, Tensor]:
    learns = learn if isinstance(learn, List) else [learn]

    # Set the dataloaders for TTA
    if dls_tta is not None:
        dls_tta = dls_tta if isinstance(dls_tta, List) else [dls_tta]
        assert len(learns) == len(dls_tta)
        for learn, dls in zip(learns, dls_tta):
            learn.dls = dls

    df_acc = None
    df_all = []

    for learn in learns:
        idx = 0 if sname == 'train' else 1
        if n_tta < 0:  # Dummy local testing:
            preds, targs = torch.rand((2, 2 if is_neg else 4)), torch.Tensor([0, 1])
        elif n_tta > 0:
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

        df_acc = df if df_acc is None else df + df_acc
        df_all.append(df)

    df = df_acc / len(learns)
    # df = df[const.VOCAB_NEG if is_neg else const.VOCAB_SHORT]  # Put columns in the right order
    df.to_csv(str(save_dir/sname) + ".csv")
    with open(save_dir/f"{sname}_all", "wb") as f:
        pickle.dump(df_all, f)

    return preds, targs


def make_all_dummy(p: Union[str, Path]) -> None:
    df = pd.read_csv(p, index_col=0)
    with open(str(p).replace(".csv", "_all"), "wb") as f:
        pickle.dump([df] * 5, f)


def predict_and_save_folds(
        learn_folds: List[Learner],
        model_name: str,
        n_tta: int,
        is_neg: bool,
        dls_tta: Optional[List[Any]] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Assumptions:
    - This is happening on vastai
    - We are only predicting the validation set
    """

    # Set the dataloaders for TTA
    if dls_tta is not None:
        assert len(learn_folds) == len(dls_tta)
        for learn, dls in zip(learn_folds, dls_tta):
            learn.dls = dls

    vocab = learn_folds[0].dls.vocab
    preds_ls, targs_ls, idx_ls = [], [], []
    for learn in learn_folds:
        assert learn.dls.vocab == vocab
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

    df = pd.DataFrame(preds.numpy(), columns=vocab)
    df.index = idx
    save_dir = _get_save_dir(name=f"{model_name}_{len(learn_folds)}fold", is_neg=is_neg)
    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)
    df.to_csv(str(save_dir/'valid') + ".csv")

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