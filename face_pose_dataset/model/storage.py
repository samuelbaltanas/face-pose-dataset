""" Storage related component """
import json
import logging
import os
from os import path
from typing import Tuple

import cv2
from PySide2 import QtCore

from face_pose_dataset.core import EstimationData, Position

PATH = "/home/sam/Desktop/sample/"


mutex = QtCore.QMutex()


class DatasetModel:
    def __init__(self, shape):
        self.identity = None
        self.path = None
        self.id_path = None

        self.data = {}

    def add_identity(self, identity):
        assert self.path is not None
        self.identity = identity
        self.id_path = path.join(self.path, self.identity)

        if not os.path.isdir(self.id_path):
            os.mkdir(self.id_path)

        # TODO: Else condition

        logging.debug("Creating new folder: %s", self.id_path)

    def save_image(self, key: Position, data: EstimationData):
        d = {}
        d["roll"] = float(data.angle.roll)
        d["pitch"] = float(data.angle.pitch)
        d["yaw"] = float(data.angle.yaw)
        d["bbox"] = data.box[:4].astype(int).tolist()

        rgb_image = _path_gen(key, False)
        d["rgb_image"] = rgb_image

        if data.depth is not None:
            depth_image = _path_gen(key, True)
            d["depth_image"] = depth_image

        self.data[key] = d

        mutex.lock()
        cv2.imwrite(path.join(self.id_path, rgb_image), data.rgb)
        if data.depth is not None:
            cv2.imwrite(path.join(self.id_path, depth_image), data.depth)
        mutex.unlock()

    def dump_data(self):
        if self.id_path is not None:
            fpath = path.join(self.id_path, "record-{}.json".format(self.identity))
            with open(fpath, "w") as f:
                logging.debug(
                    "Saving info of individual %s in %s", self.identity, fpath
                )
                json.dump(list(self.data.values()), f)


def _path_gen(key: Tuple[int, int], depth=False):
    if not depth:
        res = "image_{}-{}_rgb.png".format(key[0], key[1])
    else:
        res = "image_{}-{}_depth.png".format(key[0], key[1])

    return res
