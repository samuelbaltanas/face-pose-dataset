import os
from os import path

import cv2
import pandas as pd
from tqdm.autonotebook import tqdm

from . import biwi


def read_selection(label, index):
    head = pd.read_csv(os.path.join(biwi.BIWI_DIR, "{}.csv".format(label)))
    head = head[head.folder == index].reset_index(drop=True)

    return list(head.frame)


def eval_on_biwi(store_file, results_fol, aux_path=None):

    # TODO Load estimators

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
            image = cv2.imread(dat.path)

            # TODO Do inference

            # TODO Compute error

            results["image_id"].append(dat.identity)
            results["image_frame"].append(dat.frame)

            results["roll"].append(dat.angle[0])
            results["pitch"].append(dat.angle[1])
            results["yaw"].append(dat.angle[2])

            # TODO Store ground truth
            # TODO Store error

            pbar.update(1)

    data = pd.DataFrame(results)
    data.to_pickle(path.join(results_fol, "results.pkl"))
