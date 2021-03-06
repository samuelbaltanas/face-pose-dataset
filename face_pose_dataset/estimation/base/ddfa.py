import logging
from os import path

import cv2
import numpy as np
import pkg_resources
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import face_pose_dataset as fpdata
from face_pose_dataset import core
from face_pose_dataset.estimation import interface
from face_pose_dataset.third_party.ddfa.mobilenet_v1 import mobilenet_1 as mobilenet
from face_pose_dataset.third_party.ddfa.utils import estimate_pose, inference

__all__ = ["DdfaEstimator"]


# MODEL_PATH = path.join(
#     fpdata.PROJECT_ROOT, "data", "3ddfa-configs", "phase1_wpdc_vdc.pth.tar"
# )

MODEL_PATH = pkg_resources.resource_stream(
    "face_pose_dataset", "data/3ddfa-configs/phase1_wpdc_vdc.pth.tar"
)

if torch.cuda.is_available():
    cudnn.benchmark = True  # type: ignore


class DdfaEstimator(interface.Estimator):
    def __init__(
        self, checkpoint=MODEL_PATH, gpu=-1,
    ):
        logging.debug("[3DDFA] Loading...")
        # Gpu and cpu compatibility as per Pytorch guidelines in:
        # https://pytorch.org/docs/stable/notes/cuda.html#device-agnostic-code
        self.device = torch.device(
            "cuda:{}".format(gpu) if torch.cuda.is_available() and gpu >= 0 else "cpu"
        )
        logging.info("[3DDFA] Running on device %s", self.device)
        logging.info("[3DDFA] Loading snapshot...")
        self.checkpoint = torch.load(
            checkpoint, map_location=lambda storage, loc: storage
        )["state_dict"]
        self.model = mobilenet(
            num_classes=62
        )  # 62 = 12 (pose) + 40 (shape) +10 (expression)

        model_dict = self.model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in self.checkpoint.keys():
            model_dict[k.replace("module.", "")] = self.checkpoint[k]
        self.model.load_state_dict(model_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # DONE: Test transforms to use builtin methods.
        # ORIGINAL
        # self.transform = transforms.Compose(
        #     [ddfa.ToTensorGjz(), ddfa.NormalizeGjz(mean=127.5, std=128)]
        # )
        self.transformations = transforms.Compose(
            [
                # transforms.Scale(224),
                transforms.Lambda(lambda x: x.transpose((2, 0, 1))),
                transforms.Lambda(torch.from_numpy),
                transforms.Lambda(lambda x: x.float()),
                transforms.Normalize(
                    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]
                ),
            ]
        )

        self.gpu = gpu
        logging.info("[3DDFA] Loaded.")

    def preprocess_image(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        roi_box = inference.parse_roi_box_from_bbox(bbox)
        frame = inference.crop_img(frame, roi_box)
        frame = cv2.resize(frame, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)

        return frame

    def run(self, frame: np.ndarray) -> core.Angle:
        frame = self.transformations(frame).unsqueeze(0)
        with torch.no_grad():
            frame = frame.to(self.device)
            param = self.model(frame)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        _, ang = estimate_pose.parse_pose(param)
        yaw, pitch, roll = np.rad2deg(ang)

        return core.Angle(yaw=yaw, pitch=-pitch, roll=roll)
