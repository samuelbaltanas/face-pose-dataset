import os
import unittest

import cv2

import face_pose_dataset as fpdat
from face_pose_dataset.third_party import FSA_net


class FaceModelTest(unittest.TestCase):
    def setUp(self):
        self.detector = FSA_net.SSDDetector()
        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )
        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)
        self.estimator = FSA_net.FSAEstimator()

    def test_run(self):
        res = self.detector.run(self.sample_image)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape[:2], (1, 1))
        self.assertEqual(res.shape[3], 7)

        det = FSA_net.extract_faces(
            self.sample_image, res, self.estimator.img_size, threshold=0.8, margin=0.0
        )
        self.assertIsNotNone(res)
        self.assertEqual(
            det.shape[1:], (self.estimator.img_size, self.estimator.img_size, 3)
        )

        ang = self.estimator.run(det)
        self.assertIsNotNone(ang)
        self.assertEqual(ang.shape, (det.shape[0], 3))
