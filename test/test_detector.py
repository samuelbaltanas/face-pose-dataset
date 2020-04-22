import os
import unittest

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

import face_pose_dataset as fpdat
from face_pose_dataset.estimation import mtcnn

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


class MTCNNTest(unittest.TestCase):
    def setUp(self):
        self.detector = mtcnn.MTCNN()
        self.sample_image = cv2.imread(
            os.path.join(fpdat.PROJECT_ROOT, "data", "test", "Mark_Zuckerberg.jpg")
        )
        self.assertIsNotNone(self.sample_image, "Test image not found.")
        self.assertEqual(self.sample_image.shape[2], 3)

    def test_run(self):
        res = self.detector.run(self.sample_image, threshold=0.8,)
        self.assertIsNotNone(res)
        self.assertGreater(len(res), 0)

        det = [
            mtcnn.extract_face(
                self.sample_image, *r, output_size=(112, 112), margin=0.0
            )
            for r in mtcnn.result_iterator(*res)
        ]

        self.assertIsNotNone(res)
        self.assertGreater(len(det), 0)
        self.assertEqual(det[0].shape, (112, 112, 3))
        plt.imshow(det[0])
        plt.show()
