import sys
from time import time

import cv2
import numpy as np
import pyopenpose as op
from primesense import openni2
from rgbdMap import mapToRGBD

import consts as c
from frame import Frame


def putTextOnImg(img):
    cv2.putText(img, "FPS: " + "{:.3f}".format(fps), (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    for i in range(humanNumber):
        for j in range(c.keypointsNumber):
            if datum.poseKeypoints[i][j][2] > c.keypointThreshold:
                text = c.poses[np.argmax(posesScores[i])] + ", score = " + "{:.2f}".format(max(posesScores[i]))
                x = int(datum.poseKeypoints[i][j][0] - 10)
                y = int(datum.poseKeypoints[i][j][1] - 5)
                (textW, textH) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)[0]
                cv2.rectangle(img, (x, y + 2), (x + textW + 2, y - textH - 2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                break


# starting OpenPose
params = dict()
params["model_folder"] = "externals/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# starting frames proceed
frames = Frame()
frameTime = time()

# starting astra stream
dev = openni2.Device.open_file(sys.argv[1])
depthStream = dev.create_depth_stream()
colorStream = dev.create_color_stream()
depthStream.start()
colorStream.start()

print("Color frames number = " + str(colorStream.get_number_of_frames()))
print("Depth frames number = " + str(depthStream.get_number_of_frames()))

while True:
    frameDepth = depthStream.read_frame()
    frameColor = colorStream.read_frame()
    frameD = np.array((frameDepth.height, frameDepth.width), dtype=np.unit16, buffer=frameDepth.get_buffer_as_uint16())
    frameRGB = np.array((frameColor.height, frameColor.width, 3), dtype=np.uint8, buffer=frameColor.get_buffer_as_uint8()) / 255

    datum = op.Datum()
    c.frameHeight, c.frameWidth, _ = frameRGB.shape
    c.depthHeight, c.depthWidth, _ = frameD.shape
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop([datum])

    # array of people with keypoints is in datum.poseKeypoints
    # proceedFrame gives for every human classified pose with score [pose, score]
    posesScores = frames.proceedFrame(mapToRGBD(datum.poseKeypoints, frameD))
    humanNumber = len(datum.poseKeypoints)
    fps = 1.0 / (time() - frameTime)
    frameTime = time()
    image = datum.cvOutputData  # image is frame with drawn skeleton
    putTextOnImg(image)  # we put text with classified poses
    cv2.imshow("Human Pose Classification", image)
    cv2.waitKey(0)
