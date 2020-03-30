import logging
import os

import cv2
import numpy as np
from keras.layers import Average

import face_pose_dataset as fpdata
from face_pose_dataset.third_party.FSA_net import FSANET_model

__all__ = ["FSAEstimator", "SSDDetector", "extract_faces"]


class SSDDetector:
    def __init__(self):

        # load our serialized face detector from disk
        logging.debug("Loading face detector.")
        protoPath = os.path.join(
            fpdata.PROJECT_ROOT, "models", "face_detector", "deploy.prototxt"
        )
        modelPath = os.path.join(
            fpdata.PROJECT_ROOT,
            "models",
            "face_detector",
            "res10_300x300_ssd_iter_140000.caffemodel",
        )
        logging.debug("Face detector loaded.")
        self.net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    def run(self, input_img: np.ndarray) -> np.ndarray:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(input_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detected = self.net.forward()

        return detected


def extract_faces(
    input_image: np.ndarray,
    detection: np.ndarray,
    shape: int = 300,
    threshold: float = 0.8,
    margin: float = 0.6,
    normalize: bool = False,
):
    """ Extract detections from image. """
    detection = detection[:, :, detection[0, 0, :, 2] >= threshold, :]
    faces = None

    if detection.shape[2] > 0:
        faces = np.empty((detection.shape[2], shape, shape, 3), dtype=np.int)

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
                input_image[val[4]:val[6] + 1, val[3]:val[5] + 1, :],
                (shape, shape),
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


class FSAEstimator:
    def __init__(self, img_size=64):
        self.img_size = img_size

        # Parameters
        num_capsule = 3
        dim_capsule = 16
        routings = 2
        stage_num = [3, 3, 3]
        lambda_d = 1
        num_classes = 3
        image_size = 64
        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model1 = FSANET_model.FSA_net_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()
        model2 = FSANET_model.FSA_net_Var_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()

        num_primcaps = 8 * 8 * 3
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model3 = FSANET_model.FSA_net_noS_Capsule(
            image_size, num_classes, stage_num, lambda_d, S_set
        )()

        logging.debug("Loading models ...")

        weight_file1 = os.path.join(
            fpdata.PROJECT_ROOT,
            "models/FSA_300W_LP_model/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5",
        )
        model1.load_weights(weight_file1)
        logging.debug("Finished loading model 1.")

        weight_file2 = os.path.join(
            fpdata.PROJECT_ROOT,
            "models/FSA_300W_LP_model/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5",
        )
        model2.load_weights(weight_file2)
        logging.debug("Finished loading model 2.")

        weight_file3 = os.path.join(
            fpdata.PROJECT_ROOT,
            "models/FSA_300W_LP_model/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5",
        )
        model3.load_weights(weight_file3)
        logging.debug("Finished loading model 3.")

        inputs = FSANET_model.Input(shape=(64, 64, 3))
        x1 = model1(inputs)  # 1x1
        x2 = model2(inputs)  # var
        x3 = model3(inputs)  # w/o
        avg_model = Average()([x1, x2, x3])
        self.model = FSANET_model.Model(inputs=inputs, outputs=avg_model)

    def run(self, input_images: np.ndarray) -> np.ndarray:
        p_result = self.model.predict(input_images)

        return p_result


if __name__ == "__main__":
    pass
