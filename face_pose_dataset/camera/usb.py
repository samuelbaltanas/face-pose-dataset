import cv2

__all__ = ["VideoCamera"]


class VideoCamera(object):
    def __init__(self):
        self.video = None

    def __enter__(self):
        self.video = cv2.VideoCapture(0)
        if self.video is None:
            raise Exception("Abort. No camera available.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.video.release()
        self.video = None

    def read_rgb(self):
        success, frame = self.video.read()
        while not success:
            success, frame = self.video.read()

        # DONE: RGB/BGR discrepancy between Astra and opencv cameras
        frame = cv2.cvtColor(frame.astype("uint8"), cv2.COLOR_BGR2RGB)

        return frame

    def read_both(self):
        return self.read_rgb(), None
