import math
import os
from os import path

import cv2
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

from face_pose_dataset import core, estimation
from face_pose_dataset.datasets.biwi import biwi


def read_selection(label, index):
    head = pd.read_csv(os.path.join(biwi.BIWI_DIR, "{}.csv".format(label)))
    head = head[head.folder == index].reset_index(drop=True)

    return list(head.frame_num)


def eval_on_biwi(estimator_name: str, results_file="results.pkl"):

    # DONE Load estimators. Delegated to a method dict.
    detector = estimation.MTCNN()
    estimator: estimation.Estimator = estimation.ESTIMATORS[estimator_name]()

    dataset = biwi.BiwiDataset()

    results = {
        "image_id": [],
        "image_frame": [],
        "roll": [],
        "yaw": [],
        "pitch": [],
        "true_roll": [],
        "true_pitch": [],
        "true_yaw": [],
        "error": [],
    }

    with tqdm(total=len(dataset)) as pbar:
        dat: biwi.BiwiDatum
        for ctr, dat in enumerate(dataset):
            frame = cv2.imread(dat.path)

            # Default values for estimation if detector fails.
            est_angle = core.Angle(np.NaN, np.NaN, np.NaN)
            err = np.NaN

            true_angle = dat.angle

            det = detector.run(frame, threshold=0.8)
            if det is not None:
                bboxes, _ = det
                match = biwi.match_detection(bboxes, dat.center3d)

                # Check if any match is found
                if match >= 0:
                    # DONE Do inference
                    face = estimator.preprocess_image(frame, bboxes[match])
                    est_angle = estimator.run(face)

                    # DONE Compute error
                    err = math.sqrt(
                        (true_angle.pitch - est_angle.pitch) ** 2
                        + (true_angle.yaw - est_angle.yaw) ** 2
                    )

            results["image_id"].append(dat.identity)
            results["image_frame"].append(dat.frame_num)

            results["roll"].append(est_angle.roll)
            results["pitch"].append(est_angle.pitch)
            results["yaw"].append(est_angle.yaw)

            # TODO Store ground truth
            results["true_pitch"].append(true_angle.pitch)
            results["true_roll"].append(true_angle.roll)
            results["true_yaw"].append(true_angle.yaw)

            # TODO Store error
            results["error"].append(err)

            pbar.update(1)

    data = pd.DataFrame(results)
    data.to_pickle(results_file)


if __name__ == "__main__":
    method = "multi"
    ROOT = "/home/sam/Workspace/projects/4-ImageGathering/face_pose_dataset/data/evaluation/{}f-results.pkl".format(
        method
    )
    eval_on_biwi(method, ROOT)
