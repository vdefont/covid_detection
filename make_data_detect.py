import pandas as pd
import numpy as np
from typing import NamedTuple, Optional
from pathlib import Path
import json
from typing import List, Iterable
import shutil
import pickle

import const


class Frame(NamedTuple):
    W: float
    H: float


class Box(NamedTuple):
    X: float
    Y: float
    W: float
    H: float
    C: Optional[float] = None


def scale_bbox(frame: Frame, new_frame: Frame, box: Box) -> Box:
    # Get the scale factors
    scale_w = frame.W / new_frame.W
    scale_h = frame.H / new_frame.H

    # Scale the center
    center_x = box.X + box.W / 2
    center_y = box.Y + box.H / 2
    scaled_center_x = center_x / scale_w
    scaled_center_y = center_y / scale_h

    # Parameterize scaled box
    scaled_w = box.W / scale_w
    scaled_h = box.H / scale_h
    scaled_x = scaled_center_x - scaled_w / 2
    scaled_y = scaled_center_y - scaled_h / 2

    return Box(
        X=scaled_x,
        Y=scaled_y,
        W=scaled_w,
        H=scaled_h,
        C=box.C,
    )


def scale_bbox_pad(frame: Frame, new_frame: Frame, box: Box) -> Box:
    """
    This assumes that we processed the images using the "pad" method
    frame is not necessarily square!
    """
    dim = max(frame.W, frame.H)
    start_x = (dim - frame.W) // 2
    start_y = (dim - frame.H) // 2
    pad_box = Box(
        X=box.X + start_x,
        Y=box.Y + start_y,
        W=box.W,
        H=box.H,
        C=box.C,
    )
    return scale_bbox(frame=Frame(W=dim, H=dim), new_frame=new_frame, box=pad_box)


def _cut_overflow(frame: Frame, box: Box, thresh: float = 0.25) -> Optional[Box]:
    """
    Remove parts of the box that fall outside the frame
    If the fraction of the box that falls outside of the frame exceeds
      `thresh`, then return None
    """
    box_x, box_y, box_w, box_h = box.X, box.Y, box.W, box.H
    # Cut x and y (and adjust width + height accordingly if we do)
    if box_x < 0:
        box_w += box_x
        box_x = 0
    if box_y < 0:
        box_h += box_y
        box_y = 0
    # Cut width and height
    over_right = box_x + box_w - frame.W
    if over_right > 0:
        box_w -= over_right
    over_btm = box_y + box_h - frame.H
    if over_btm > 0:
        box_h -= over_btm
    # Check if fraction cut exceeds thresh
    area_orig = box.H * box.W
    area_new = box_h * box_w
    if area_orig == 0:
        return None
    cut = (area_orig - area_new) / area_orig
    if cut > thresh:
        return None
    return Box(X=box_x, Y=box_y, W=box_w, H=box_h, C=box.C)


def test_cut_overflow() -> None:
    frame = Frame(W=5, H=3)
    box = Box(X=0, Y=0, H=4, W=1)
    # Expect: 0, 0, 3, 1
    print(_cut_overflow(frame, box, 1.))
    print(_cut_overflow(frame, box, 0.25))
    # Expect: None
    print(_cut_overflow(frame, box, 0.24))

    box = Box(X=1,Y=2,W=2,H=3,C=5)
    # Expect: 1, 2, 2, 1, 5
    print(_cut_overflow(frame, box, 0.67))
    # Expect: None
    print(_cut_overflow(frame, box, 0.66))

    box = Box(X=-1,Y=-1,W=10,H=10)
    # Expect: (0, 0, 5, 3)
    print(_cut_overflow(frame, box, 1.))

    box = Box(X=1,Y=-1,W=3,H=5)
    # Expect: (1, 0, 3, 3)
    print(_cut_overflow(frame, box, 1.))

    box = Box(X=-1,Y=1,W=7,H=3)
    # Expect: 0, 1, 5, 2
    print(_cut_overflow(frame, box, 1.))


def unscale_bbox_pad(frame: Frame, new_frame: Frame, box: Box) -> Optional[Box]:
    """
    frame: The shrunk frame (eg. 224x224)
    new_frame: The original frame (eg. 1092 x 1492)
    """
    # Step 1: scale up to the longest edge
    dim = max(new_frame.H, new_frame.W)
    box = scale_bbox(frame=frame, new_frame=Frame(H=dim, W=dim), box=box)

    # Step 2: cut down to a rectangle
    start_x = (dim - new_frame.W) // 2
    start_y = (dim - new_frame.H) // 2
    box_x = box.X - start_x
    box_y = box.Y - start_y

    # Step 3: deal with overflow
    box = Box(X=box_x, Y=box_y, W=box.W, H=box.H, C=box.C)
    return _cut_overflow(frame=new_frame, box=box)


def test_unscale_bbox_pad() -> None:
    frame_small = Frame(W=8, H=8)
    frame_orig = Frame(W=16, H=32)

    def _test(box: Box) -> None:
        print(unscale_bbox_pad(frame=frame_small, new_frame=frame_orig, box=box))

    _test(Box(X=1,Y=1,W=4,H=3,C=8))
    # Expect: 0, 4, 12, 12

    _test(Box(X=1, Y=1, W=3, H=3, C=8))
    # Expect: None

    _test(Box(X=3, Y=3, W=2, H=2, C=1))
    # Expect: 4, 12, 8, 8

    _test(Box(X=3, Y=2, W=4, H=6))
    _test(Box(X=3, Y=2, W=3, H=7))
    # Expect: 4, 8, 12, 24
    _test(Box(X=3, Y=2, W=4, H=7))
    # Expect: None

    print("\n\nSAME THING, TRANSPOSED\n\n")

    frame_small = Frame(W=8, H=8)
    frame_orig = Frame(W=32, H=16)

    def _test(box: Box) -> None:
        print(unscale_bbox_pad(frame=frame_small, new_frame=frame_orig, box=box))

    _test(Box(X=1, Y=1, W=3, H=4, C=8))
    # Expect: 4, 0, 12, 12

    _test(Box(X=1, Y=1, W=3, H=3, C=8))
    # Expect: None

    _test(Box(X=3, Y=3, W=2, H=2, C=1))
    # Expect: 12, 4, 8, 8

    _test(Box(X=2, Y=3, W=6, H=4))
    _test(Box(X=2, Y=3, W=7, H=3))
    # Expect: 8, 4, 24, 12
    _test(Box(X=2, Y=3, W=7, H=4))
    # Expect: None


def test_bbox_pad() -> None:
    frame_orig = Frame(8, 4)
    frame_small = Frame(4, 4)
    def _test_scale_unscale(box):
        box_new = scale_bbox_pad(frame=frame_orig, new_frame=frame_small, box=box)
        print(box_new)
        # Expect original
        print(unscale_bbox_pad(frame=frame_small, new_frame=frame_orig, box=box_new))

    _test_scale_unscale(Box(X=2, Y=1, W=2, H=3, C=3))
    # Expect: X=1, Y=1.5, W=1, H=1.5

    _test_scale_unscale(Box(X=5,Y=1,W=2,H=3,C=0.5))


def scale_bbox_ls(frame: Frame, new_frame: Frame, boxes: Iterable[Box], pad: bool = True) -> List[Box]:
    scale_func = scale_bbox_pad if pad else scale_bbox
    return [scale_func(frame, new_frame, box) for box in boxes]


def test_scale_bbox() -> None:
    frame = Frame(W=20, H=10)
    box = Box(X=1, Y=1, W=18, H=8)
    frame_new = Frame(W=10, H=5)

    print(scale_bbox(frame, frame_new, box))
    print("Expect: Box(0.5, 0.5, 9, 4)\n")

    box = Box(X=16, Y=5, W=4, H=5)
    print(scale_bbox(frame, frame_new, box))
    print('Expect: Box(X=8.0, Y=2.5, W=2.0, H=2.5)\n')

    frame = Frame(2, 1)
    frame_new = Frame(1, 2)
    box = Box(X=0.5, Y=0.2, W=1, H=0.3)
    print(scale_bbox_ls(frame, frame_new, [box, box], pad=False))
    print('Expect: Box(X=0.25, Y=0.4, W=0.5, H=0.6)\n')

    # Pad method
    box = Box(X=2, Y=1, W=2, H=3, C=3)
    print(scale_bbox_ls(frame=Frame(8, 4), new_frame=Frame(4, 4), boxes=[box]))
    # Expect: X=1, Y=1.5, W=1, H=1.5


def unscale_bbox_ls(frame: Frame, new_frame: Frame, boxes: Iterable[Box], pad: bool = True) -> List[Box]:
    scale_func = unscale_bbox_pad if pad else scale_bbox
    ret = [scale_func(frame=frame, new_frame=new_frame, box=box) for box in boxes]
    ret = [r for r in ret if r is not None]
    return ret


def test_unscale_bbox_ls():
    frame_small = Frame(W=8, H=8)
    frame_orig = Frame(W=32, H=16)
    boxes = [
        Box(X=1, Y=1, W=3, H=4, C=8),
        Box(X=1, Y=1, W=3, H=3, C=8),
        Box(X=3, Y=3, W=2, H=2, C=1),
        Box(X=2, Y=3, W=6, H=4),
        Box(X=2, Y=3, W=7, H=3),
        Box(X=2, Y=3, W=7, H=4),
    ]
    for b in unscale_bbox_ls(frame=frame_small, new_frame=frame_orig, boxes=boxes, pad=True):
        print(b)
    # Expect:
    # 4, 0, 12, 12
    # 12, 4, 8, 8
    # 8, 4, 24, 12
    # 8, 4, 24, 12

    print("\n\nNO PAD\n\n")

    frame_small = Frame(W=8, H=8)
    frame_orig = Frame(W=16, H=32)
    for b in unscale_bbox_ls(frame=frame_small, new_frame=frame_orig, boxes=boxes, pad=False):
        print(b)
    # Expect:
    # 2, 4, 6, 16
    # 2, 4, 6, 12
    # 6, 12, 4, 8
    # 4, 12, 12, 16
    # 4, 12, 14, 12
    # 4, 12, 14, 16


def make_boxes(row: pd.Series, frame_new: Frame) -> List[Box]:
    """
    Make a list of rescaled Box objects from a row in image_data
    """
    js = json.loads(row.boxes)
    boxes = [Box(X=e['x'], Y=e['y'], W=e['width'], H=e['height']) for e in js]
    frame_old = Frame(W=row.width, H=row.height)
    return scale_bbox_ls(frame=frame_old, new_frame=frame_new, boxes=boxes)


def make_object_xml(box: Box) -> str:
    return f"""
        <object>
                <name>opacity</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{box.X}</xmin>
                        <ymin>{box.Y}</ymin>
                        <xmax>{box.X+box.W}</xmax>
                        <ymax>{box.Y+box.H}</ymax>
                </bndbox>
        </object>"""


def make_annotation_xml(path: Path, objects: Iterable[str], frame_new: Frame) -> str:
    objs = '\n'.join(objects)
    return f"""<annotation>
        <folder>{str(path.parent.name)}</folder>
        <filename>{str(path.name)}</filename>
        <path>{str(path)}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{int(frame_new.W)}</width>
                <height>{int(frame_new.H)}</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>{objs}
</annotation>
    """.replace('\n\n', '\n')


def make_xml(path: Path, boxes: Iterable[Box], frame_new: Frame) -> str:
    """
    For a list of boxes corresponding to some image path, make VOC-formatted XML
    """
    objects = map(make_object_xml, boxes)
    return make_annotation_xml(path=path, objects=objects, frame_new=frame_new)


def create_box_data(frame_new: Frame, src_dir_str: str, dst_dir_str: str, extn: str, test_only: bool = False) -> None:
    """
    Create XML annotations for all rows in image data. Also copy the corresponding
    images from src_dir to dst_dir, to fit the expected folder structure
    """
    src_dir = const.subdir_data_class(path=True) / src_dir_str
    dst_dir = const.subdir_data_detect(path=True) / dst_dir_str
    dst_dir_boxes = const.subdir_data_detect(path=True) / (dst_dir_str + "_boxes")

    image_data = None
    if test_only is False:
        image_data = pd.read_csv(const.subdir_data_csv(path=True) / "train_image_level_prep.csv")

    # Make dirs
    dst_dir.mkdir(parents=True)
    for s in ['train', ('test' if test_only else 'valid')]:
        (dst_dir / s).mkdir()
        (dst_dir / s / 'annotations').mkdir()
        (dst_dir / s / 'images').mkdir()
    if not test_only:
        dst_dir_boxes.mkdir()

    for s in ['test'] if test_only else ['train', 'valid']:
        id_boxes = {}
        for path in (src_dir/s).glob(f'**/*{extn}'):
            id = path.name.replace(f".{extn}", "")
            new_path = dst_dir/s/'images'/path.name

            if s == 'test':
                boxes = [Box(X=0, Y=0, W=1, H=1)]
            else:
                rows = image_data[image_data.id == id]
                row = rows.iloc[0]
                if pd.isna(row.boxes):
                    continue
                boxes = make_boxes(row=row, frame_new=frame_new)
                id_boxes[id] = boxes

            annotation = make_xml(path=new_path, boxes=boxes, frame_new=frame_new)
            shutil.copy(path, new_path) # Copy image
            with open(dst_dir/s/'annotations'/f'{id}.xml', 'w') as f:
                f.write(annotation)

        if not test_only:
            with open(dst_dir_boxes/s, "wb") as f:
                pickle.dump(id_boxes, f)

    # Must copy some images to train set for dataloader to work
    if test_only:
        for i, p in enumerate((dst_dir/'test'/'images').glob('*')):
            if i == 70:
                break
            shutil.copy(p, dst_dir/'train'/'images'/p.name)
            xml_name = p.name.replace(f'.{extn}', '.xml')
            shutil.copy(dst_dir/'test'/'annotations'/xml_name, dst_dir/'train'/'annotations'/xml_name)


def make_boxes_png_224(test_only: bool = False):
    create_box_data(
        frame_new=Frame(W=224, H=224),
        src_dir_str="png224_test" if test_only else "png224",
        dst_dir_str="png224_test" if test_only else "png224",
        extn='png',
        test_only=test_only,
    )