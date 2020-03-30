import unittest

from face_pose_dataset import core, pose_storage


class AngleEstimationTest(unittest.TestCase):
    def setUp(self):
        # PARAMS
        self.dims = 10, 10
        self.yaw_range = pose_storage.DEFAULT_YAW_RANGE
        self.pitch_range = pose_storage.DEFAULT_PITCH_RANGE

        self.storage = pose_storage.ScoreMatrix(
            self.dims, pitch_range=self.pitch_range, yaw_range=self.yaw_range
        )

    def test_ranges(self):
        self.assertEqual(len(self.storage.y_range), self.storage.shape[0])
        self.assertEqual(len(self.storage.x_range), self.storage.shape[1])

    def test_creation(self):
        self.assertEqual(self.storage.scores.shape, self.dims)
        self.assertEqual(self.storage.scores.shape, self.storage.shape)

    def aux_angle_estimation(self, pos, expected_angle):
        angle = self.storage.convert_position(pos)
        self.assertEqual(angle.yaw, expected_angle.yaw)
        self.assertEqual(angle.pitch, expected_angle.pitch)
        self.assertEqual(angle.roll, 0.0)

        return angle

    def test_lower_angle_estimation(self):
        self.aux_angle_estimation((0, 0), core.Angle(0.0, -180.0 + 18.0, -90.0 + 9.0))

    def test_middle_angle_estimation(self):
        self.aux_angle_estimation((5, 5), core.Angle(0.0, 18.0, 9.0))

    def test_upper_angle_estimation(self):
        self.aux_angle_estimation((9, 9), core.Angle(0.0, 180.0 - 18.0, 90.0 - 9.0))


class PoseEstimationTest(unittest.TestCase):
    def setUp(self):
        # PARAMS
        self.dims = 10, 10
        self.yaw_range = pose_storage.DEFAULT_YAW_RANGE
        self.pitch_range = pose_storage.DEFAULT_PITCH_RANGE

        self.storage = pose_storage.ScoreMatrix(
            self.dims, pitch_range=self.pitch_range, yaw_range=self.yaw_range
        )

        # Fill scores of dummy values
        for pitch in range(10):
            for yaw in range(10):
                self.storage[pitch, yaw] = yaw * 10 + pitch

    def aux_pose_estimation(self, angle, expected_pos):
        pos = self.storage.locate_angle(angle)

        self.assertEqual(pos, expected_pos)
        self.assertEqual(self.storage[pos], self.storage[expected_pos])
        return pos

    def test_lower_pose_estimation(self):
        angle = core.Angle(0.0, -162.0, -81.0)
        self.aux_pose_estimation(angle, (0, 0))

    def test_upper_pose_estimation(self):
        angle = core.Angle(0.0, 162.0, 81.0)
        self.aux_pose_estimation(angle, (9, 9))

    def test_avg_pose_estimation(self):
        angle = core.Angle(0.0, -18.0, -9.0)
        self.aux_pose_estimation(angle, (4, 4))

    def test_edge_pose_estimation(self):
        angle = core.Angle(0.0, 0.0, 0.0)
        pos = self.aux_pose_estimation(angle, (5, 5))

        angle = core.Angle(0.0, 18.0, 9.0)
        ref_pos = self.storage.locate_angle(angle)

        self.assertEqual(pos, ref_pos)


if __name__ == "__main__":
    unittest.main()
