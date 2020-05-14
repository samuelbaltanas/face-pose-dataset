import logging
import os
from typing import Tuple

import cv2
import numpy as np
import pkg_resources
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Average

import face_pose_dataset as fpdata
from face_pose_dataset import core
from face_pose_dataset.estimation import interface, mtcnn
from face_pose_dataset.third_party.fsa_estimator import FSANET_model

# tf.disable_v2_behavior()


__all__ = ["FSAEstimator", "SSDDetector"]


class SSDDetector:
    def __init__(self):

        # load our serialized face detector from disk
        logging.debug("Loading face detector.")
        proto_path = os.path.join(
            fpdata.PROJECT_ROOT, "data", "face_detector", "deploy.prototxt"
        )
        model_path = os.path.join(
            fpdata.PROJECT_ROOT,
            "data",
            "face_detector",
            "res10_300x300_ssd_iter_140000.caffemodel",
        )
        logging.debug("Face detector loaded.")
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def run(self, input_img: np.ndarray, threshold=0.8) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(input_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detected = self.net.forward()

        detected = detected[:, :, detected[0, 0, :, 2] >= threshold, :]

        return detected


def extract_faces(
    input_image: np.ndarray,
    detection: np.ndarray,
    shape: Tuple[int, int] = (300, 300),
    threshold: float = 0.8,
    margin: float = 0.6,
    normalize: bool = False,
):
    """ Extract detections from image. """
    faces = None

    if detection.shape[2] > 0:
        faces = np.empty((detection.shape[2], *shape, 3), dtype=np.int)

        (h0, w0) = input_image.shape[:2]
        detection[0, 0, :, 3:7] *= np.array([w0, h0, w0, h0])

        if margin > 0:
            w = detection[0, 0, :, 5] - detection[0, 0, :, 3]
            h = detection[0, 0, :, 6] - detection[0, 0, :, 4]

            detection[0, 0, :, 3] -= w * margin
            detection[0, 0, :, 4] -= h * margin
            detection[0, 0, :, 5] += w * margin
            detection[0, 0, :, 6] += h * margin

            del w, h

        detection = detection.astype(np.int)

        detection[0, 0, :, 3] = np.clip(detection[0, 0, :, 3], 0, input_image.shape[1])
        detection[0, 0, :, 5] = np.clip(detection[0, 0, :, 5], 0, input_image.shape[1])
        detection[0, 0, :, 4] = np.clip(detection[0, 0, :, 4], 0, input_image.shape[0])
        detection[0, 0, :, 6] = np.clip(detection[0, 0, :, 6], 0, input_image.shape[0])

        for idx, val in enumerate(detection[0, 0]):
            faces[idx, :, :, :] = cv2.resize(
                input_image[val[4] : val[6] + 1, val[3] : val[5] + 1, :], shape,
            )
            if normalize:
                faces[idx, :, :, :] = cv2.normalize(
                    faces[idx, :, :, :],
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                )

    return faces


class FSAEstimator(interface.Estimator):
    def __init__(self, use_gpu=False):
        self.img_size = 64, 64

        # Parameters
        num_capsule = 3
        dim_capsule = 16
        routings = 2
        stage_num = [3, 3, 3]
        lambda_d = 1
        num_classes = 3
        num_primcaps = 7 * 3
        m_dim = 5
        s_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        gpus_available = tf.config.list_physical_devices(device_type="GPU")

        logging.info("[FSANET] GPUs available: %s.", gpus_available)
        if use_gpu and gpus_available:
            config = tf.ConfigProto(
                log_device_placement=False,  # logging.getLogger().level < logging.INFO
            )
            config.gpu_options.allow_growth = True
            logging.info("[FSANET] Set on GPU.")
        else:
            config = tf.ConfigProto(
                log_device_placement=False,  # logging.getLogger().level < logging.INFO,
                device_count={"CPU": 1, "GPU": 0},
            )
            logging.info("[FSANET] Set on CPU.")
        self.session = tf.InteractiveSession(config=config)

        tf.compat.v1.keras.backend.set_session(self.session)
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            model1 = FSANET_model.FSA_net_Capsule(
                self.img_size[0], num_classes, stage_num, lambda_d, s_set
            )()
            model2 = FSANET_model.FSA_net_Var_Capsule(
                self.img_size[0], num_classes, stage_num, lambda_d, s_set
            )()

            num_primcaps = 8 * 8 * 3
            s_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

            model3 = FSANET_model.FSA_net_noS_Capsule(
                self.img_size[0], num_classes, stage_num, lambda_d, s_set
            )()

            logging.info("[FSANET] Loading data ...")

            weight_file1 = pkg_resources.resource_filename(
                "face_pose_dataset",
                "data/FSA_300W_LP_model/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5",
            )
            model1.load_weights(weight_file1)
            logging.info("[FSANET] Model 1 loaded.")

            weight_file2 = pkg_resources.resource_filename(
                "face_pose_dataset",
                "data/FSA_300W_LP_model/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5",
            )
            model2.load_weights(weight_file2)
            logging.info("[FSANET] Model 2 loaded.")

            weight_file3 = pkg_resources.resource_filename(
                "face_pose_dataset",
                "data/FSA_300W_LP_model/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5",
            )
            model3.load_weights(weight_file3)
            logging.info("[FSANET] Model 3 loaded.")

            inputs = FSANET_model.Input(shape=(*self.img_size, 3))
            x1 = model1(inputs)  # 1x1
            x2 = model2(inputs)  # var
            x3 = model3(inputs)  # w/o
            avg_model = Average()([x1, x2, x3])

            self.model = FSANET_model.Model(inputs=inputs, outputs=avg_model)

            logging.info("[FSANET] Loaded.")

    def preprocess_image(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        # DONE Test 0.4 margin instead of 0.6
        res = mtcnn.extract_face(
            frame, (64, 64), bbox, landmark=None, margin=0.6, normalize=True
        )
        return np.expand_dims(res, 0)

    def run(self, input_images: np.ndarray) -> np.ndarray:
        with self.graph.as_default():
            tf.keras.backend.set_session(self.session)
            yaw, pitch, roll = self.model.predict(input_images)[0]
            ang = core.Angle(yaw=yaw, pitch=pitch, roll=-roll)
        return ang


if __name__ == "__main__":
    pass
