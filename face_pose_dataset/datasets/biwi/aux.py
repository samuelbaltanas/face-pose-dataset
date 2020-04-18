import cv2
import numpy as np

# DONE Move harcoded values to auxiliary
IDENTITY_LABELS = np.array(
    [
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "M01",
        "M02",
        "M03",
        "M04",
        "M05",
        "M06",
        "M07",
        "M08",
        "F03",
        "M09",
        "M10",
        "F05",
        "M11",
        "M12",
        "F02",
        "M01",
        "M13",
        "M14",
    ],
    dtype=str,
)
FEATURE_SEX = np.array(
    [
        "F",
        "F",
        "F",
        "F",
        "F",
        "F",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "M",
        "F",
        "M",
        "M",
        "F",
        "M",
        "M",
        "M",
    ],
    dtype=str,
)

REPEATED = {15: 3, 18: 5, 21: 2, 22: 7}

# Configuration of the depth camera in the BIWI dataset
RVEC = np.array([[517.679, 0, 320], [0, 517.679, 240.5], [0, 0, 1]])
CAMERA_MATRIX = np.eye(3)
TVEC = np.zeros((3, 1))
DIST_COEFFS = 0, 0, 0, 0


def project_points(points):
    """ Convenience method in place of cv2.projectPoints.
        It uses the hardcoded parameters of the BIWI dataset's camera.
    """
    return cv2.projectPoints(points, RVEC, TVEC, CAMERA_MATRIX, DIST_COEFFS)[0]
