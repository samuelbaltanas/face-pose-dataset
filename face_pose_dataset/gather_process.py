from typing import Tuple

import numpy as np

from face_pose_dataset import core, pose_storage

DEFAULT_THRESHOLD = 0.5


class CellHandler:
    """ Class dedicated to gather images for a specific cell. """

    def __init__(
        self,
        pos: Tuple[int, int],
        storage: pose_storage.ScoreMatrix,
        threshold=DEFAULT_THRESHOLD,
    ):
        self._pos = pos
        self._storage = storage
        self._threshold = threshold
        raise NotImplementedError()

    def add_image(self, angle: core.Angle, image: np.ndarray):
        """ Checks if an image should be stored, then buffers it. """
        raise NotImplementedError()

    def check_continue(self) -> bool:
        raise NotImplementedError()


class DomainIterator:
    """ Class used to traverse the AngleContainer, gathering one image per cell.

    TODO: Implement missing functionality. maybe use https://scipy-lectures.org/advanced/advanced_python/index.html#bidirectional-communication
    """

    def __init__(self, score_matrix: np.ndarray):
        assert len(score_matrix.shape) == 2
        self.score_matrix = score_matrix
        self.current_pos = 0, self.score_matrix.shape[1] - 1
        self.right_flag = (self.score_matrix.shape[1] - 1) % 2
        self.handle = CellHandler(self.current_pos, self.score_matrix)

    def __next__(self) -> CellHandler:
        if self.handle.check_continue():
            self._increase_counter()
            self.handle = CellHandler(
                self.current_pos, self.score_matrix[self.current_pos]
            )
            return self.handle
        raise StopIteration

    def _increase_counter(self):
        """ Increase counter in a left-to-right, bottom-up fashion. """
        assert len(self.current_pos) == 2

        x, y = self.current_pos
        x_max, y_max = self.score_matrix.shape

        move = 1 if x % 2 == self.right_flag else -1
        if 0 < x + move < x_max:
            x += move
        elif 0 < y - 1:
            y -= 1

    def add_image(self, image: np.ndarray, angle: core.Angle):
        """ Checks if an image should be stored, then buffers it. """
        raise NotImplementedError()

    def check_continue(self) -> bool:
        raise NotImplementedError()
