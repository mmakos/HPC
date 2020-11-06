import pyopenpose as op
import numpy as np
import cv2

import astraPython as ap
from rgbdMap import mapToRGBD
from frame import Frame
from time import time
import consts as c
import poses

def putTextOnImg( img ):
    cv2.putText( img, "FPS: " + "{:.3f}".format( fps ), ( 5, 23 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ( 255, 0, 0 ), 2 )
    for i in range( humanNumber ):
        for j in range( c.keypointsNumber ):
            if datum.poseKeypoints[ i ][ j ][ 2 ] > c.keypointThreshold:
                text = poses.poses[ np.argmax( posesScores[ i ] ) ] + ", score = " + "{:.2f}".format( max( posesScores[ i ] ) )
                x = int( datum.poseKeypoints[ i ][ j ][ 0 ] - 10 )
                y = int( datum.poseKeypoints[ i ][ j ][ 1 ] - 5 )
                ( textW, textH ) = cv2.getTextSize(  text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, thickness = 2 )[ 0 ]
                cv2.rectangle( img, ( x, y + 2 ), ( x + textW + 2, y - textH - 2), ( 0, 255, 0 ), cv2.FILLED )
                cv2.putText( img, text, ( x, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 255, 255, 255 ), 2 )
                break


# starting OpenPose
params = dict()
params[ "model_folder" ] = "../../../externals/openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure( params )
opWrapper.start()

# starting frames proceed
frames = Frame()
frameTime = time()

# starting astra stream
stream = ap.AstraStream()
stream.initialize()

while True:
    datum = op.Datum()
    stream.proceedFrame()
    frameRGB = np.array( stream.getTestRGB() ) / 255
    frameD = np.array( stream.getTestDepth() )

    frameRGB = cv2.imread( '../../../img/ladies.jpg' )          #TODO to delete
    c.frameHeight, c.frameWidth, dim = frameRGB.shape
    frameD = np.ones( ( c.frameHeight, c.frameWidth ), int )    #TODO to delete
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop( [ datum ] )

    # array of people with keypoints is in datum.poseKeypoints
    # proceedFrame gives for every human classified pose with score [pose, score]
    posesScores = frames.proceedFrame( mapToRGBD( datum.poseKeypoints, frameD ) )
    humanNumber = len( datum.poseKeypoints )
    fps = 1.0 / ( time() - frameTime )
    frameTime = time()
    image = datum.cvOutputData    # image is frame with drawn skeleton
    putTextOnImg( image )                          # we put text with classified poses
    cv2.imshow( "Human Pose Classification", image )
    cv2.waitKey( 0 )
