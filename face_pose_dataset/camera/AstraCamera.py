from typing import Optional

import numpy as np
from openni import _openni2 as c_api
from openni import openni2

__all__ = ["AstraCamera", "PROPERTIES"]

PROPERTIES = {"width": 640, "height": 480, "fps": 30}


class AstraCamera:
    def __init__(self):
        self.dev: Optional[openni2.Device] = None
        self.video_stream: Optional[openni2.VideoStream] = None
        self.depth_stream: Optional[openni2.VideoStream] = None

    def _start_depth_stream(self) -> openni2.VideoStream:
        if self.dev is not None:
            depth_stream = self.dev.create_depth_stream()
        else:
            raise Exception("Initialize AstraCamera using with statement.")

        if depth_stream is None:
            raise Exception("Current device has not a depth sensor.")

        depth_stream.start()
        depth_stream.set_video_mode(
            c_api.OniVideoMode(
                pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM,
                resolutionX=PROPERTIES["width"],
                resolutionY=PROPERTIES["height"],
                fps=PROPERTIES["fps"],
            )
        )

        self.depth_stream = depth_stream

        return depth_stream

    def _start_rgb_stream(self) -> openni2.VideoStream:
        if self.dev is not None:
            rgb_stream = self.dev.create_color_stream()
        else:
            raise Exception("Initialize AstraCamera using with statement.")

        if rgb_stream is None:
            raise Exception("Current device has not a color sensor.")

        rgb_stream.set_video_mode(
            openni2.c_api.OniVideoMode(
                pixelFormat=openni2.c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                resolutionX=PROPERTIES["width"],
                resolutionY=PROPERTIES["height"],
                fps=PROPERTIES["fps"],
            )
        )
        rgb_stream.set_mirroring_enabled(False)
        rgb_stream.start()

        self.video_stream = rgb_stream

        return rgb_stream

    def read_depth(self):
        if self.depth_stream is None:
            self._start_depth_stream()
        frame = self.depth_stream.read_frame()
        frame_data = frame.get_buffer_as_uint16()
        depth_array = np.ndarray(
            (PROPERTIES["height"], PROPERTIES["width"]),
            dtype=np.uint16,
            buffer=frame_data,
        )

        return depth_array

    def read_rgb(self):
        if self.video_stream is None:
            self._start_rgb_stream()
        frame = self.video_stream.read_frame()
        frame_data = frame.get_buffer_as_uint8()
        rgb_array = np.ndarray(
            (PROPERTIES["height"], PROPERTIES["width"], 3),
            dtype=np.uint8,
            buffer=frame_data,
        )

        return rgb_array

    def __enter__(self):
        openni2.initialize("/home/sam/OpenNI-Linux-x64-2.3/Redist/")
        self.dev = openni2.Device(None,)  # same as Device.open_any()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.video_stream is not None:
            self.video_stream.stop()
        else:
            self.video_stream = None
        if self.depth_stream is not None:
            self.depth_stream.stop()
        else:
            self.depth_stream = None
        openni2.unload()


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    with AstraCamera() as cam:
        input_img = cam.read_rgb()
        fig, ax = plt.subplots()
        webcam_preview = ax.imshow(input_img)
        fig.show()

        print("Start plotting camera ...")

        while True:
            # get video frame
            input_img = cam.read_rgb()
            ax.imshow(input_img)
            fig.canvas.draw()
            plt.pause(1 / 60)
