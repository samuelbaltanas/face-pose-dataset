from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from face_pose_dataset import core
from face_pose_dataset.third_party.hopenet import hopenet

from . import interface, mtcnn

cudnn.enabled = True  # type: ignore

__all__ = ["HopenetEstimator"]


class HopenetEstimator(interface.Estimator):
    def __init__(
        self,
        snapshot_path="/home/sam/Workspace/exploration/deep-head-pose/models/hopenet_alpha1.pkl",
        gpu=0,
    ):
        # ResNet50 structure
        self.model = hopenet.Hopenet(
            torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66
        )

        print("Loading snapshot.")
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path)
        self.model.load_state_dict(saved_state_dict)

        self.transformations = transforms.Compose(
            [
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.gpu = gpu
        self.model.cuda(self.gpu)

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.idx_tensor = list(range(66))
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda(self.gpu)

    def preprocess_image(self, frame, bbox):
        res = mtcnn.extract_face(
            frame, (224, 224), bbox, margin=(0.4, 0.7, 0.4, 0.1), normalize=False
        )
        # TODO: Test if color conversion is needed.
        # res = cv2.cvtColor(res.astype("uint8"), cv2.COLOR_BGR2RGB)
        return res

    def run(self, input_images):
        img = Image.fromarray(input_images)

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(self.gpu)

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
            pitch=-pitch_predicted.item(),
            roll=roll_predicted.item(),
        )

    @property
    def img_size(self) -> Tuple[int, int]:
        return 224, 224
