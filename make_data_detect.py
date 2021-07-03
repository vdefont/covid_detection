import pandas as pd
import numpy as np
from typing import NamedTuple, Optional
from pathlib import Path
import json
from typing import List, Iterable
import shutil

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


def scale_bbox_ls(frame: Frame, new_frame: Frame, boxes: Iterable[Box]) -> List[Box]:
    return [scale_bbox(frame, new_frame, box) for box in boxes]


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
    print(scale_bbox_ls(frame, frame_new, [box, box]))
    print('Expect: Box(X=0.25, Y=0.4, W=0.5, H=0.6)\n')


def make_boxes(row: pd.Series, frame_new: Frame) -> List[Box]:
    """
    Make a list of rescaled Box objects from a row in image_data
    """
    js = json.loads(row.boxes)
    boxes = [Box(X=e['x'],Y=e['y'],W=e['width'],H=e['height']) for e in js]
    frame_old = Frame(W=row.width, H=row.height)
    return scale_bbox_ls(frame_old, frame_new, boxes)


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


def create_box_data(frame_new: Frame, src_dir_str: str, dst_dir_str: str, test_only: bool = False) -> None:
    """
    Create XML annotations for all rows in image data. Also copy the corresponding
    images from src_dir to dst_dir, to fit the expected folder structure
    """
    src_dir = const.subdir_data_class(path=True) / src_dir_str
    dst_dir = const.subdir_data_detect(path=True) / dst_dir_str

    image_data = None
    if test_only is False:
        image_data = pd.read_csv(const.subdir_data_csv(path=True) / "train_image_level_prep.csv")

    # Make dirs
    dst_dir.mkdir(parents=True)
    for s in ['train', ('test' if test_only else 'valid')]:
        (dst_dir / s).mkdir()
        (dst_dir / s / 'annotations').mkdir()
        (dst_dir / s / 'images').mkdir()

    for s in ['test'] if test_only else ['train', 'valid']:
        for path in (src_dir/s).glob('**/*png'):
            id = path.name.replace(".png", "")
            new_path = dst_dir/s/'images'/path.name

            if s == 'test':
                boxes = [Box(X=0, Y=0, W=1, H=1)]
            else:
                rows = image_data[image_data.id == id]
                row = rows.iloc[0]
                if pd.isna(row.boxes):
                    continue
                boxes = make_boxes(row=row, frame_new=frame_new)

            annotation = make_xml(path=new_path, boxes=boxes, frame_new=frame_new)
            shutil.copy(path, new_path) # Copy image
            with open(dst_dir/s/'annotations'/f'{id}.xml', 'w') as f:
                f.write(annotation)

    # Must copy some images to train set for dataloader to work
    if test_only:
        for i, p in enumerate((dst_dir/'test'/'images').glob('*')):
            if i == 70:
                break
            shutil.copy(p, dst_dir/'train'/'images'/p.name)
            xml_name = p.name.replace('.png', '.xml')
            shutil.copy(dst_dir/'test'/'annotations'/xml_name, dst_dir/'train'/'annotations'/xml_name)


def make_boxes_png_224(test_only: bool = False):
    create_box_data(
        frame_new=Frame(W=224, H=224),
        src_dir_str="png224_test" if test_only else "png224",
        dst_dir_str="png224_test" if test_only else "png224",
        test_only=test_only,
    )