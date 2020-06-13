import pyopenpose as op
import cv2

from src.rgbdMap import mapToRGBD
from src.frame import Frame
from time import time
import src.consts as c
import src.poses


# TODO - must return frame from RGBD video stream
# frame should be RGB image in cv2 format and separate depth canal
def getVideoFrame():
    return [[[]]], [[]]


def displayImg():
    cv2.imshow( "Human Pose Classification", image )
    for i in range( humanNumber ):
        for j in range( c.keypointsNumber ):
            if datum.poseKeypoints[ i ][ j ][ 2 ] > c.keypointThreshold:
                cv2.putText( image, src.poses.poses( poses[ 0 ] ) + ", score = " + src.poses.poses( poses[ 1 ] ),
                             ( int( datum.poseKeypoints[ i ][ j ][ 0 ] ),
                               int( datum.poseKeypoints[ i ][ j ][ 1 ] ) + 10 ),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 255, 0 ), 2 )
                break
    cv2.putText( image, "FPS: " + str( fps ), ( 5, 5 ), cv2.FONT_HERSHEY_SIMPLEX, 1, ( 0, 255, 0 ), 2,
                 bottomLeftOrigin = False )


# starting OpenPose
params = dict()
params[ "model_folder" ] = "../../models/"
opWrapper = op.WrapperPython()
opWrapper.configure( params )
opWrapper.start()

# starting frames proceed
frames = Frame()
frameTime = time()

while True:
    datum = op.Datum()
    frameRGB, frameD = getVideoFrame()
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop( [ datum ] )

    # array of people with keypoints is in datum.poseKeypoints
    # proceedFrame gives for every human classified pose with score [pose, score]
    poses = frames.proceedFrame( mapToRGBD( datum.poseKeypoints ) )
    humanNumber = len( datum.poseKeypoints )
    fps = 1.0 / (time() - frameTime)
    frameTime = time()
    image = datum.cvInputData[ :, :, : ]

    displayImg()
