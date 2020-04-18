import os
import unittest

import cv2
import tensorflow as tf

import face_pose_dataset as fpdat
from face_pose_dataset.estimation import ddfa as ddfa
from face_pose_dataset.estimation import fsanet, hopenet, mtcnn

# DONE: Use Sandberg MTCNN


gpus = tf.config.experimental.list_physical_devices("GPU")

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def _common_estimation(case, image, detector, estimator):
    res = detector.run(image)
    case.assertIsNotNone(res)
    case.assertGreater(len(res), 0)

    det = mtcnn.extract_faces(
        image, res, estimator.img_size, margin=(0.4, 0.7, 0.4, 0.1)
    )

    import matplotlib.pyplot as plt

    plt.imshow(det[0])
    plt.show()

    case.assertIsNotNone(det)
    case.assertGreater(len(det), 0)
    case.assertEqual(det[0].shape, (*estimator.img_size, 3))

    ang = estimator.run(det[0])
    case.assertIsNotNone(ang)
    print(ang)


class SSDFSATest(unittest.TestCase):
    def setUp(self):
        self.detector = fsanet.SSDDetector()
        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )
        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)
        self.estimator = fsanet.FSAEstimator()

    def test_run(self):
        res = self.detector.run(self.sample_image)
        self.assertIsNotNone(res)
        self.assertEqual(res.shape[:2], (1, 1))
        self.assertEqual(res.shape[3], 7)

        det = fsanet.extract_faces(
            self.sample_image, res, self.estimator.img_size, threshold=0.2, margin=0.0
        )
        self.assertIsNotNone(res)
        self.assertEqual(det.shape[1:], (*self.estimator.img_size, 3))

        ang = self.estimator.run(det)
        self.assertIsNotNone(ang)
        self.assertEqual(ang.shape, (det.shape[0], 3))
        print(ang)


class MtcnnFSATest(unittest.TestCase):
    def setUp(self):
        self.detector = mtcnn.MTCNN()
        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )
        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)
        self.estimator = fsanet.FSAEstimator()

    def test_run(self):
        res = self.detector.run(self.sample_image, threshold=0.8,)
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)

        det = mtcnn.extract_faces(self.sample_image, res, self.estimator.img_size,)

        self.assertIsNotNone(res)
        self.assertGreater(len(det), 0)
        self.assertEqual(det[0].shape, (*self.estimator.img_size, 3))

        ang = self.estimator.run(det)
        self.assertIsNotNone(ang)
        print(ang)


class HopeTest(unittest.TestCase):
    def setUp(self):
        self.detector = mtcnn.MTCNN()
        self.estimator = hopenet.HopenetEstimator()

        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )

        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)

    def test_run(self):
        res = self.detector.run(self.sample_image, threshold=0.8)
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)

        det = mtcnn.extract_faces(
            self.sample_image, res, self.estimator.img_size, margin=(0.4, 0.7, 0.4, 0.1)
        )

        import matplotlib.pyplot as plt

        plt.imshow(det[0])
        plt.show()

        self.assertIsNotNone(det)
        self.assertGreater(len(det), 0)
        self.assertEqual(det[0].shape, (*self.estimator.img_size, 3))

        ang = self.estimator.run(det[0])
        self.assertIsNotNone(ang)
        print(ang)


class DDFATest(unittest.TestCase):
    def setUp(self):
        self.detector = mtcnn.MTCNN()
        self.estimator = ddfa.DdfaEstimator()

        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )
        self.sample_image = cv2.cvtColor(self.sample_image, cv2.COLOR_BGR2RGB)
        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)

    def test_run(self):
        res = self.detector.run(self.sample_image, threshold=0.8)
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)

        det = mtcnn.extract_faces(
            self.sample_image,
            res,
            self.estimator.img_size,  # margin=(0.4, 0.7, 0.4, 0.1)
        )

        import matplotlib.pyplot as plt

        plt.imshow(det[0])
        plt.show()

        self.assertIsNotNone(det)
        self.assertGreater(len(det), 0)
        self.assertEqual(det[0].shape, (*self.estimator.img_size, 3))

        ang = self.estimator.run(det[0])
        self.assertIsNotNone(ang)
        print(ang)
