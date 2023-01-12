import argparse
import os
import sys
from datetime import datetime

import cv2
import keyboard
import numpy as np
from primesense import _openni2 as c_api
from primesense import openni2

sys.path.insert(1, '../func')
import hpc.core.display as display


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default=datetime.now().strftime("%Y%m%d%H%M%S"), help="Path to recorded video relative to /data/video.")
    parser.add_argument("-c", "--color_preview", action="store_true")
    parser.add_argument("-d", "--depth_preview", action="store_true")
    return parser.parse_known_args()[0]


def recordOni():
    depthStream = dev.create_depth_stream()
    colorStream = dev.create_color_stream()
    depthStream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                  resolutionX=640, resolutionY=480, fps=30))
    colorStream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                  resolutionX=1280, resolutionY=720, fps=30))
    dev.set_image_registration_mode(True)
    dev.set_depth_color_sync_enabled(True)
    depthStream.set_mirroring_enabled(True)
    colorStream.set_mirroring_enabled(True)

    depthStream.start()
    colorStream.start()

    recorder = openni2.Recorder((path + ".oni").encode('utf-8'))
    recorder.attach(depthStream)
    recorder.attach(colorStream)

    print("Press 's' to start recording")
    keyboard.wait("s")
    print()

    recorder.start()

    print("Recording...\nPress 'q' to stop recording")
    keyboard.wait("q")
    print()

    recorder.stop()
    depthStream.stop()
    colorStream.stop()


def recordRs():
    global path
    path = path + "/"
    if not os.path.isdir(path):
        os.mkdir(path)
    print("Press 's' to start recording")

    rec = False
    i = 0
    while True:
        frames = dev.wait_for_frames()
        frameDepth = np.asanyarray(frames.get_depth_frame().get_data())
        frameColor = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()), cv2.COLOR_BGR2RGB)

        dt_string = datetime.now().strftime("%d%m%Y_%H%M%S%f")
        if args.color_preview:
            display.displaySmallFrameNumber(frameColor, i)
            cv2.imshow("Color frame", frameColor)
        if args.depth_preview:
            cv2.imshow("Depth frame", frameDepth)
        if rec:
            cv2.imwrite(path + "/" + dt_string + "_d.png", frameDepth)
            cv2.imwrite(path + "/" + dt_string + "_c.jpg", frameColor)
            i = i + 1

        if not rec and cv2.waitKey(1) & 0xFF == ord('s'):
            rec = True
            print("Recording...\nPress 'q' to stop recording")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    args = getArgs()
    path = "../../data/videos/" + args.video
    try:
        print("Trying to open Openni device...")
        if sys.platform == "win32":
            openni2.initialize(
                "../../externals/OpenNI/Windows/Astra OpenNI2 Development Instruction(x64)_V1.3/OpenNI2/OpenNI-Windows-x64-2.3.0.63/Redist")
        else:
            openni2.initialize("../../OpenNI/Linux/OpenNI-Linux-x64-2.3.0.63/Redist")
        dev = openni2.Device.open_any()
        print("Openni device opened.")
        recordOni()
        openni2.unload()
    except:
        print("No openni device found.")
        print("Trying to open RealSense device...")
        try:
            import pyrealsense2 as rs

            dev = rs.pipeline()
            dev.start()
            print("RealSense device opened.")
            recordRs()
        except:
            print("No RealSense device found.")

    print("Recording ended.")
