import numpy as np
import cv2
import sys
from primesense import openni2
from primesense import _openni2 as c_api

import consts as c


# starting astra stream

openni2.initialize(
            "../../OpenNI/Windows/Astra OpenNI2 Development Instruction(x64)_V1.3/OpenNI2/OpenNI-Windows-x64-2.3.0.63/Redist" )
dev = openni2.Device.open_any()
colorStream = dev.create_color_stream()
colorStream.start()

print( "Color frames number = " + str( colorStream.get_number_of_frames() ) )

while True:
    frameColor = colorStream.read_frame()
    frameRGB = np.array( ( frameColor.height, frameColor.width, 3 ), dtype=np.uint8, buffer=frameColor.get_buffer_as_uint8() ) / 255

    c.frameHeight, c.frameWidth, dim = frameRGB.shape

    cv2.imshow( "Human Pose Classification", frameRGB )
    cv2.waitKey( 0 )
