from copy import deepcopy
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

import const


def get_x_y():
    base = const.subdir_data_class(path=True) / "jpg224_3fold"
    paths = list(base.glob('**/*jpg'))
    y = [p.parent.name for p in paths]
    imgs = [np.array(Image.open(p)) for p in paths]
    X = np.array(imgs)
    X = X.reshape((X.shape[0], -1))

    return X, y


def pca_coords(X):
    p = PCA(2)
    x2 = p.fit_transform(X)
    return x2


def tsne_coords(X):
    t = TSNE(2)
    return t.fit_transform(X)


def get_df(x2, y) -> pd.DataFrame:
    df = pd.DataFrame(x2, columns=["x0", "x1"])
    df["y"] = y
    return df


def plot(df: pd.DataFrame, sample: int) -> None:
    rows = random.sample(range(df.shape[0]), sample)
    df = df.iloc[rows]

    colormap = {
        'neg': 'green', 'typ': 'red', 'ind': 'yellow', 'atyp': 'blue'
    }
    df['c'] = df.y.map(colormap)

    df.plot.scatter(x="x0", y="x1", c="c")
    plt.show()


# dft = get_df(x2t, y)
# plot(dft, 1000)


from faimed3d.faimed3d.all import *

dicom_paths = list((const.dir_original_data(True) / 'train').glob('**/*dcm'))

td = TensorDicom3D.create(dicom_paths[0])
show_image_3d(td)