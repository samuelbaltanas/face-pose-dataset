import logging
from os import path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pkg_resources
from skimage import transform as trans
import tensorflow as tf

import face_pose_dataset as fpdata
from face_pose_dataset.third_party.mtcnn_sandberg import mtcnn_keras, mtcnn_tensorflow

_FEATURE_TRANSFORM = np.array(
    [
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ],
    dtype=np.float32,
)

MODEL_PATH = pkg_resources.resource_filename(
    "face_pose_dataset", "data/mtcnn_tensorflow/"
)

Margin = Union[int, float]
MarginDescriptor = Union[
    Margin,
    Tuple[int, int],
    Tuple[float, float],
    Tuple[int, int, int, int],
    Tuple[float, float, float, float],
]


def parse_margins(
    margin: MarginDescriptor, width: int = 1, height: int = 1,
) -> Tuple[int, int, int, int]:

    if isinstance(margin, (int, float)):
        # Case single value: uniform margin
        expanded_margin: Tuple[Margin, Margin, Margin, Margin] = (
            margin,
            margin,
            margin,
            margin,
        )
    elif isinstance(margin, tuple) and len(margin) == 2:
        # Two elements: (w, h) margin
        expanded_margin = (
            margin[0],
            margin[1],
            margin[0],
            margin[1],
        )
    elif isinstance(margin, tuple) and len(margin) == 4:
        # margin per dimension (w0, h0, w1, h1)
        expanded_margin = margin  # type: ignore
    else:
        raise TypeError(
            "The margin parameter only accepts values of type %s." % MarginDescriptor
        )

    if isinstance(expanded_margin[0], float):
        res: Tuple[int, int, int, int] = (
            int(expanded_margin[0] * width),
            int(expanded_margin[1] * height),
            int(expanded_margin[2] * width),
            int(expanded_margin[3] * height),
        )
    elif isinstance(expanded_margin[0], int):
        res = expanded_margin
    else:
        raise TypeError(
            "The margin parameter only accepts values of type %s." % MarginDescriptor
        )

    return res


def extract_face(
    frame: np.ndarray,
    output_size: Tuple[int, int],
    bbox: Optional[np.ndarray] = None,
    landmark: Optional[np.ndarray] = None,
    normalize=True,
    margin: MarginDescriptor = 0,
) -> np.ndarray:
    """ General method to extract a face from a bounding box.

        :param margin: Margin added to the tight bounding box.
            See parse_margins documentation for more info about padding format,
        :param normalize: Perform MINMAX normalization on the RGB values.
        :param output_size: Target output size after adding the margin.
        :param bbox: Bounding box in coordinates of frame. Format: [LEFT, TOP, RIGHT, BOTTOM].
        :param frame: Original image.
        :param landmark: 5 face keypoints used for alignment.
            Leave it as None to perform normal face extraction.

        :return: Cropped face image.


    """
    ret = None

    if frame.shape[2] != 3:
        raise Exception(
            "The `align` function only works in single images. Shape of `source_image`: {} != (?, ?, 3)".format(
                frame.shape
            )
        )

    if landmark is not None:
        target_features = _FEATURE_TRANSFORM
        source_features = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(
            source_features, target_features,
        )
        M = tform.params[0:2, :]

        # M = cv2.estimateAffine2D(
        #     target_features.reshape((1, 5, 2)),
        #     source_features.reshape((1, 5, 2)),
        # )[0]

        ret = cv2.warpAffine(
            frame, M, (output_size[1], output_size[0]), borderValue=0.0
        )
    else:

        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(frame.shape[1])
            det[1] = int(frame.shape[0])
            det[2] = frame.shape[1] - det[0]
            det[3] = frame.shape[0] - det[1]
        else:
            det = bbox

        bb = np.zeros(4, dtype=np.int32)

        w = abs(det[2] - det[0])
        h = abs(det[3] - det[1])

        margin = parse_margins(margin, width=w, height=h)

        bb[0] = np.clip(det[0] - margin[0], 0, frame.shape[1])
        bb[1] = np.clip(det[1] - margin[1], 0, frame.shape[0])
        bb[2] = np.clip(det[2] + margin[2], 0, frame.shape[1])
        bb[3] = np.clip(det[3] + margin[3], 0, frame.shape[0])

        ret = frame[bb[1] : bb[3], bb[0] : bb[2], :]

        if output_size is not None:
            ret = cv2.resize(ret, (output_size[1], output_size[0]))

        if normalize:
            ret = cv2.normalize(
                ret, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
            )

    return ret


def extract_faces(
    input_image: List[np.ndarray],
    detection: Tuple[List[np.ndarray], np.ndarray],
    shape: Tuple[int, int] = (300, 300),
    margin: Union[float, Tuple[float, float], Tuple[float, float, float, float]] = 0.6,
    normalize: bool = False,
):
    faces = None

    bbs, points = detection

    if len(bbs) > 0:
        faces = np.empty((len(bbs), *shape, 3), dtype=np.int)

        for idx, bb in enumerate(bbs):
            faces[idx, :, :, :] = extract_face(
                input_image, shape, bb, normalize=normalize, margin=margin,
            )

    return faces


class MTCNN:

    min_size = 20
    threshold = [0.6, 0.6, 0.7]
    factor = 0.709

    def __init__(
        self, model_root=MODEL_PATH, gpu=0
    ):
        # DONE: Changes to work with tensorflow v2
        logging.info("[MTCNN] Loading.")
        self.pnet, self.rnet, self.onet = mtcnn_keras.generate_models(model_root, False)
        logging.info("[MTCNN] Loaded.")

    def run(
        self, image: np.ndarray, threshold=0.5,
    ) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:

        bbs, points = mtcnn_tensorflow.detect_face(
            image,
            MTCNN.min_size,
            self.pnet,
            self.rnet,
            self.onet,
            MTCNN.threshold,
            MTCNN.factor,
        )

        bbx = []
        pts = []

        for idx, bb in enumerate(bbs):
            if bb[4] >= threshold:
                bbx.append(bb)
                pts.append(points[:, idx])

        if not bbx:
            return None
        else:
            return bbx, pts
