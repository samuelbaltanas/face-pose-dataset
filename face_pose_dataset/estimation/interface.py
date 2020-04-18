import abc

import numpy as np

from face_pose_dataset import core


class Estimator(abc.ABC):
    def run(self, frame: np.ndarray) -> core.Angle:
        raise NotImplementedError("Estimator is an abstract class")

    def preprocess_image(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """ Proceed with the necessary image preprocessing before inference.

         Parameters
         ----------
         bbox
            Bounding box corresponding to a single image.

         frame
            Original image in RGB format.

         Returns
         ---------
         res
            Extracted image containing an individuals face.
            Image preprocessing and margin details are delegated to each subclass.

        """
        raise NotImplementedError("Estimator is an abstract class")

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
