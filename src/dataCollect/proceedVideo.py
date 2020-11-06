import argparse
from time import time

import numpy as np
import cv2
import sys
import os
import consts as c
import poses
from primesense import openni2

from frame import Frame
from rgbdMap import mapToRGBD

dir_path = os.path.dirname( os.path.realpath( __file__ ) )
sys.path.append( dir_path + '/../../externals/openpose/build/python/openpose/Release' )
os.environ[ 'PATH' ] = os.environ[
                           'PATH' ] + ';' + dir_path + '/../../externals/openpose/build/x64/Release;' + dir_path + '/../../externals/openpose/build/bin;'
import pyopenpose as op


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "videoPath", type = str, help = "Path to file you want to proceed" )
    parser.add_argument( "-d", "--depth", help = "Depth frames will be proceeded as well", action="store_true" )
    parser.add_argument( "-v", "--view", help = "View only mode", action="store_true" )
    parser.add_argument( "-p", "--proceed", help = "Frames will be converted to skeleton image and saved in given path" )
    return parser.parse_args()


def initOpenPose():
    # starting OpenPose
    params = dict()
    params[ "model_folder" ] = "../../../externals/openpose/models/"
    wrapper = op.WrapperPython()
    wrapper.configure( params )
    wrapper.start()
    return wrapper


def initFrameDimensions():
    c.frameHeight, c.frameWidth, dim = frameRGB.shape
    if frameD is not None:
        c.depthHeight, c.depthWidth, dim = frameD.shape


def getColorFrame():
    if oni:
        frameColor = colorStream.read_frame()
        frameColor = np.array( ( frameColor.height, frameColor.width, 3 ), dtype = np.uint8,
                               buffer = frameColor.get_buffer_as_uint8() ) / 255
    else:
        ret, frameColor = vid.read()
    return frameColor


def getDepthFrame():
    if not args.depth:
        frameDepth = np.zeros( ( c.depthWidth, c.depthHeight ) )
    else:
        frameDepth = depthStream.read_frame()
        frameDepth = np.array( ( frameDepth.height, frameDepth.width ), dtype = np.uint16,
                               buffer = frameDepth.get_buffer_as_uint16() )
    return frameDepth


def proceedFrame():
    datum = op.Datum()
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop( [ datum ] )

    # array of people with keypoints is in datum.poseKeypoints
    # getSkeletons gives for every human skeleton image
    image = datum.cvOutputData  # image is frame with drawn skeleton

    # convert frame to skeleton image
    skeletons = []
    if args.proceed:
        skeletons = frame.getSkeletons( mapToRGBD( datum.poseKeypoints, frameD ) )

    return image, skeletons


if __name__ == '__main__':
    args = parseArgs()

    # OpenNI file
    if args.videoPath[ -4: ] == ".oni":
        oni = True
        print( "OpenNI file" )
        vid = openni2.Device.open_file( args.videoPath )
        colorStream = vid.create_color_stream()
        colorStream.start()
        framesNumber = colorStream.get_number_of_frames()
        if args.depth:
            depthStream = vid.create_depth_stream()
            depthStream.start()
    # Regular video
    else:
        oni = False
        print( "Regular video" )
        vid = cv2.VideoCapture( args.videoPath )
        framesNumber = int( vid.get( cv2.CAP_PROP_FRAME_COUNT ) )

    # initialise openPose and Frame class
    if not args.view:
        opWrapper = initOpenPose()
        frame = Frame()

    for i in range( framesNumber ):
        frameRGB = getColorFrame()
        frameD = getDepthFrame()
        if i is 0:
            initFrameDimensions()

        if not args.view:
            frameRGB, skeletonImages = proceedFrame()
            for j, img in skeletonImages:
                cv2.imwrite( f"{ args.proceed }/f{ i }s{ j }", img )

        cv2.imshow( "Video frame", frameRGB )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break
