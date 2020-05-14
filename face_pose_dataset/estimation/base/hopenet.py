import logging
from os import path
from typing import Tuple

import pkg_resources
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

import face_pose_dataset as fpdata
from face_pose_dataset import core
from face_pose_dataset.estimation import interface, mtcnn
from face_pose_dataset.third_party.hopenet import hopenet

if torch.cuda.is_available():
    cudnn.enabled = True  # type: ignore

__all__ = ["HopenetEstimator"]


MODEL_PATH = pkg_resources.resource_stream(
    "face_pose_dataset", "data/hopenet/hopenet_alpha1.pkl"
)


class HopenetEstimator(interface.Estimator):
    def __init__(
        self, snapshot_path=MODEL_PATH, gpu=-1,
    ):
        logging.debug("[HOPENET] Loading...")
        # Gpu and cpu compatibility as per Pytorch guidelines in:
        # https://pytorch.org/docs/stable/notes/cuda.html#device-agnostic-code
        if torch.cuda.is_available() and gpu >= 0:
            self.device = torch.device("cuda:{}".format(gpu))
        else:
            self.device = torch.device("cpu")
        logging.info("[HOPENET] Running on device %s", self.device)

        # ResNet50 structure
        self.model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66
        )
        self.model.to(self.device)

        logging.info("[HOPENET] Loading snapshot...")
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(saved_state_dict)

        self.transformations = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.idx_tensor = list(range(66))
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)
        logging.info("[HOPENET] Loaded.")

    def preprocess_image(self, frame, bbox):
        res = mtcnn.extract_face(
            frame, (224, 224), bbox, margin=(0.4, 0.7, 0.4, 0.1), normalize=False
        )
        # DONE: Test if color conversion is needed.
        # res = cv2.cvtColor(res.astype("uint8"), cv2.COLOR_BGR2RGB)
        return res

    def run(self, input_images):
        img = Image.fromarray(input_images)

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).to(self.device)

        yaw, pitch, roll = self.model(img)

        yaw_predicted = func.softmax(yaw, dim=1)
        pitch_predicted = func.softmax(pitch, dim=1)
        roll_predicted = func.softmax(roll, dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        return core.Angle(
            yaw=yaw_predicted.item(),
            pitch=pitch_predicted.item(),
            roll=-roll_predicted.item(),
        )

    @property
    def img_size(self) -> Tuple[int, int]:
        return 224, 224
