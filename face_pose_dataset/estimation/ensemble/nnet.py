import argparse
import os
import pathlib
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data

import face_pose_dataset
from face_pose_dataset import core
from face_pose_dataset.estimation import interface
from face_pose_dataset.estimation.ensemble import AverageEstimator

__all__ = ["NnetWrapper", "NnetEnsemble"]


def normalize_range(v: np.ndarray):
    """ Crop 9 vector into space """
    v[0::3] = v[0::3] / 180.0  # roll
    v[1::3] = v[1::3] / 180.0 # pitch
    v[2::3] = v[2::3] / 180.0  # yaw

    return v


def inverse_norm(v: np.ndarray):
    """ Crop 9 vector into space """
    v[0::3] = v[0::3] * 180.0  # roll
    v[1::3] = v[1::3] * 180.0  # pitch
    v[2::3] = v[2::3] * 180.0  # yaw

    return v


class NnetEnsemble(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False
        )
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument(
            "--activation",
            type=str,
            choices=["prelu", "relu", "sigmoid"],
            default="relu",
        )
        parser.add_argument(
            "--optimizer", type=str, choices=["adam", "sgd"], default="sgd"
        )
        parser.add_argument(
            "--preprocessing",
            type=str,
            choices=["identity", "normalize"],
            default="identity",
        )
        parser.add_argument(
            "--loss", type=str, choices=["mse", "huber"], default="mse"
        )
        parser.add_argument("--n_hidden_1", type=int, default=20)
        parser.add_argument("--n_hidden_2", type=int, default=20)
        return parser

    def __init__(self, hparams, out_loss = 0):
        super().__init__()
        # Hyperparameters loading
        self.hparams = hparams
        self.data_train, self.data_val = None, None

        if hparams.activation == "prelu":
            activation = nn.PReLU
        elif hparams.activation == "relu":
            activation = nn.ReLU
        elif hparams.activation == "sigmoid":
            activation = nn.Sigmoid
        else:
            raise AttributeError("Invalid activation function.")

        self.out_loss = out_loss
        out = 9 if out_loss == 1 else 3

        # Model definition
        if self.hparams.n_hidden_1 > 0:
            self.fc1 = nn.Linear(9, self.hparams.n_hidden_1)
            torch.nn.init.constant_(self.fc1.weight, 180.)# 1/self.hparams.n_hidden_1)
            self.act1 = activation()
            if self.hparams.n_hidden_2 > 0:
                self.fc2 = nn.Linear(
                    self.hparams.n_hidden_1, self.hparams.n_hidden_2
                )
                torch.nn.init.constant_(self.fc2.weight, 180. )#  1/self.hparams.n_hidden_2)
                self.act2 = activation()
                self.fc3 = nn.Linear(self.hparams.n_hidden_2, out)
            else:
                self.fc3 = nn.Linear(self.hparams.n_hidden_1, out)
        else:
            self.fc3 = nn.Linear(9, out)

        torch.nn.init.constant_(self.fc3.weight, 180.)

    def forward(self, input: Any):
        x = input
        if self.hparams.n_hidden_1 > 0:
            x = self.act1(self.fc1(input))
            if self.hparams.n_hidden_2 > 0:
                x = self.act2(self.fc2(x))
        x = self.fc3(x)


        if self.out_loss == 1:
            xx = x.clone()

            xx[:, 0::3] = F.normalize(x[:, 0::3], p=1, dim=1)
            xx[:, 1::3] = F.normalize(x[:, 1::3], p=1, dim=1)
            xx[:, 2::3] = F.normalize(x[:, 2::3], p=1, dim=1)

            res = torch.mul(input, xx)

            r = [res[:, 0::3].sum(1), res[:, 1::3].sum(1), res[:, 2::3].sum(1)]
            r = torch.stack(r, dim=1)
        else:
            r = x

        return r

CHECKPOINT_ROOT = face_pose_dataset.PROJECT_ROOT.joinpath("data", "multi_classifier", "lightning_logs")

class NnetWrapper(interface.Estimator):
    def __init__(self, checkpoint, out_loss=1):
        if isinstance(checkpoint, int):
            checkpoint = list(CHECKPOINT_ROOT.joinpath("version_%d" % checkpoint, "checkpoints").glob("*.ckpt"))[0]
        elif isinstance(checkpoint, (str, pathlib.Path)):
            pass
        else:
            raise ValueError("Checkpoint must be an integer or represent a path.")
        self.origin = AverageEstimator(activation='concat')
        self.model = NnetEnsemble.load_from_checkpoint(checkpoint, out_loss=out_loss)
        self.model.eval()

    def preprocess_image(self, *args, **kwargs) -> np.ndarray:
        return self.origin.preprocess_image(*args, **kwargs)

    def run(
        self, input_images: Tuple[np.ndarray, ...]
    ) -> Union[core.Angle, np.ndarray]:
        with torch.no_grad():
            res = self.origin.run(input_images)
            res = torch.from_numpy(res).float()
            pred = self.model.forward(res)[0].numpy()
        return core.Angle(*pred)

