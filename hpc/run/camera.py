import os
import sys

import cv2
import numpy as np

sys.path.insert(1, '../func')
import hpc.consts as c


class Camera:
    def __init__(self, video=None, noDepth=False):
        self.noDepth = noDepth
        self.vType = None
        self.colorStream = None
        self.depthStream = None
        self.vid = None
        self.videoName = video
        self.framesNumber = None
        self.__getStreams()

    def __getStreams(self):
        if not self.videoName:  # stream
            self.vType = 'rs'
            print("Opening RealSense stream...")
            try:
                import pyrealsense2 as rs
                self.vid = rs.pipeline()
                self.vid.start()
                self.framesNumber = "Infinite number of"
            except RuntimeError:
                raise FileNotFoundError("No available streams detected.")
        elif self.videoName[-4:] == ".oni":
            self.vType = 'oni'
            print("Opening OpenNI file...")
            from primesense import openni2
            self.vid = openni2.Device.open_file(self.videoName)
            self.colorStream = self.vid.create_color_stream()
            self.colorStream.start()
            self.depthStream = self.vid.create_depth_stream()
            self.depthStream.start()
            self.framesNumber = self.colorStream.get_number_of_frames()
        # RealSense video / tiago video
        elif self.videoName[-4:] == ".bag":
            self.vType = 'bag'
            print("Opening ros file (RealSense)...")
            import pyrealsense2 as rs
            self.vid = rs.pipeline()
            conf = rs.config()
            rs.config.enable_device_from_file(conf, self.videoName, repeat_playback=False)
            conf.enable_stream(rs.stream.depth)
            conf.enable_stream(rs.stream.color)
            profile = self.vid.start(conf)
            playback = profile.get_device().as_playback()
            playback.set_real_time(False)
            self.framesNumber = "Unknown quantity of"
        # RGB and depth image video (net datasets and tiago recorded to images)
        elif self.videoName[-1] == "/" or self.videoName[-1] == "\\":
            self.vType = 'img'
            print("Opening video from images...")
            self.vid = sorted(os.listdir(self.videoName))  # vid is array of file names
            self.framesNumber = len(self.vid)
            self.colorStream = 0
            self.depthStream = 1
        # Regular video
        elif self.videoName[-4:] == ".mp4" or self.videoName[-4:] == ".avi" or self.videoName[-4:] == ".mov":
            self.vType = 'reg'
            print("Opening regular video")
            self.vid = cv2.VideoCapture(self.videoName)
            self.framesNumber = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.noDepth = True
        else:
            raise TypeError("Unsupported video type.")
        print("Opened.")

    # throws EOFError when end of video
    def getFrame(self):
        if self.vType == 'bag' or self.vType == 'rs':
            try:
                frames = self.vid.wait_for_frames()
                frameDepth = np.asanyarray(frames.get_depth_frame().get_data())
                frameColor = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_BGR2RGB)
            except RuntimeError:
                raise EOFError("No more frames")
        elif self.vType == 'oni':
            frameColor = self.colorStream.read_frame()
            frameColor = np.array((frameColor.height, frameColor.width, 3), dtype=np.uint8,
                                  buffer=frameColor.get_buffer_as_uint8()) / 255
            frameDepth = self.depthStream.read_frame()
            frameDepth = np.array((frameDepth.height, frameDepth.width), dtype=np.uint16,
                                  buffer=frameDepth.get_buffer_as_uint16())
        elif self.vType == 'img':
            try:
                frameColor = cv2.imread(self.videoName + self.vid[self.colorStream])
                frameDepth = cv2.imread(self.videoName + self.vid[self.depthStream], cv2.IMREAD_ANYDEPTH)
                self.colorStream = self.colorStream + 2
                self.depthStream = self.depthStream + 2
            except IndexError:
                raise EOFError("No more frames.")
        elif self.vType == 'reg':
            frameDepth = np.zeros((c.depthHeight, c.depthWidth))
            ret, frameColor = self.vid.read()
        else:
            raise TypeError("Invalid video format.")
        if self.noDepth and self.vType != 'reg':
            frameDepth = np.zeros((c.depthHeight, c.depthWidth))
        return frameColor, frameDepth
