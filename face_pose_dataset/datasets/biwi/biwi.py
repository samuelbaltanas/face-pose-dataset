# https://github.com/natanielruiz/deep-head-pose/blob/master/code/datasets.py
# https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py

import os
from os import path
from typing import List, NamedTuple, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from face_pose_dataset.core import Angle

from . import aux

# Default paths to the dataset
BIWI_DIR = "/home/sam/Datasets/data/hpdb/"


integer = Union[int, np.integer]


class BiwiDatum(NamedTuple):
    """ Data wrapper over results. """

    path: str
    center3d: np.ndarray
    angle: Angle
    identity: int
    frame_num: int

    def read_image(self):
        """ Load frame image. """
        return cv2.imread(self.path)


def read_ground_truth(fd):
    mat = []
    with open(fd, "r") as fp:
        for line in fp:
            ll = []
            items = line.split()
            if len(items) == 0:
                continue
            for i in items:
                ll.append(float(i))
            mat.append(ll)

    m = np.array(mat)
    translation = m[3, :]
    rotation = m[:3, :]

    return translation, angle_from_matrix(rotation)


def angle_from_matrix(rot: np.ndarray) -> Angle:
    """ Transform rotation matrix to Euclidean angle. """
    rot = rot.T

    roll = -np.arctan2(rot[1, 0], rot[0][0]) * 180 / np.pi
    yaw = (
        -np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2),) * 180 / np.pi
    )
    pitch = np.arctan2(rot[2, 1], rot[2, 2]) * 180 / np.pi

    return Angle(roll, pitch, yaw)


def match_detection(
    det_data: List[np.ndarray], center: np.ndarray, margin: float = 0.5
):
    """ Use BIWI's ground truth to find which face detection should be selected.

        :param det_data: List of detections.
        :param ground_truth: Ground truth from BIWI. 3D point corresponding to the center of the face.

    """
    projected = aux.project_points(center)[0].ravel()

    best: Tuple[int, np.float] = (-1, np.infty)
    for idx, bbox in enumerate(det_data):

        diff_x = (bbox[2] - bbox[0]) * margin / 2
        diff_y = (bbox[3] - bbox[1]) * margin / 2

        bbox[0] -= diff_x
        bbox[1] -= diff_y
        bbox[2] += diff_x
        bbox[3] += diff_y

        if (
            projected[0] < bbox[0]
            or projected[1] < bbox[1]
            or projected[0] > bbox[2]
            or projected[1] > bbox[3]
        ):
            continue
        else:
            center_bb = np.array((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2))
            dist = np.sqrt(
                (projected[0] - center_bb[0]) ** 2 + (projected[1] - center_bb[1]) ** 2
            )
            if dist < best[1]:
                best = idx, dist

    return best[0]


class BiwiIndividual:
    def __init__(self, data_path=BIWI_DIR, index: integer = 1):
        self.index = index
        self.directory = path.join(data_path, "{:02d}".format(index))

    def __iter__(self):
        # Gather individual's images
        for image_file in sorted(
            fname.name
            for fname in os.scandir(self.directory)
            if fname.name.endswith(".png")
        ):
            index = int(image_file.split("_")[1])
            pose_file = path.join(self.directory, "frame_{:05d}_pose.txt".format(index))
            center3d, angle = read_ground_truth(pose_file)

            yield BiwiDatum(
                path=path.join(self.directory, image_file),
                center3d=center3d,
                angle=angle,
                identity=self.index,
                frame_num=index,
            )

    def __getitem__(self, index: integer):

        if isinstance(index, (np.integer, int)):
            index = "{:05d}".format(index)

        pose_file = path.join(self.directory, "frame_{}_pose.txt".format(index))
        image_file = path.join(self.directory, "frame_{}_rgb.png".format(index))

        if not path.isfile(image_file):
            raise IndexError("File {} does not exist.".format(image_file))

        center3d, angle = read_ground_truth(pose_file)

        return BiwiDatum(image_file, center3d, angle, self.index, int(index))


class BiwiDataset:
    """ Class wrapping the BIWI dataset.

        This class is better used as an iterable, as the __getitem__ method is not ensured to succed in the whole range of frames.
        E.g. an individual might contain frames [..., 119, 121, ...].

    """

    def __init__(self, data_path=BIWI_DIR):
        self._dir = data_path

    def __getitem__(self, key) -> Union[BiwiIndividual, BiwiDatum]:

        if not isinstance(key, tuple):
            res = BiwiIndividual(self._dir, key)
        else:
            individual = BiwiIndividual(self._dir, key[0])
            if len(key) == 1:
                res = individual
            else:
                res = individual[key[1]]

        return res

    def __iter__(self):
        for identifier in range(1, 25):
            individual = BiwiIndividual(self._dir, identifier)

            for frame in individual:
                yield frame

    def as_pandas(self):
        """ Return pandas dataframe containing all info in dataset.

            WARN: This operation is costly and should not be called often.
        """
        data = {
            "identity": [],
            "frame": [],
            "center_x": [],
            "center_y": [],
            "center_z": [],
            "roll": [],
            "pitch": [],
            "yaw": [],
        }
        with tqdm(total=self.__len__) as tq:
            for dat in self:
                data["identity"].append(dat.identity)
                data["frame"].append(dat.frame_num)
                data["center_x"].append(dat.center3d[0])
                data["center_y"].append(dat.center3d[1])
                data["center_z"].append(dat.center3d[2])
                data["roll"].append(dat.angle.roll)
                data["pitch"].append(dat.angle.pitch)
                data["yaw"].append(dat.angle.yaw)
                tq.update(1)

        return pd.DataFrame(data)

    def __len__(self):
        return 15678
