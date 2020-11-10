import argparse
import numpy as np
import cv2
import sys
import os
from primesense import openni2
from time import time

os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
import tensorflow as tf

sys.path.insert( 1, '../func' )
import display
import consts as c
from frame import Frame
from rgbdMap import mapToRGBD

dir_path = os.path.dirname( os.path.realpath( __file__ ) )
sys.path.append( dir_path + '/../../externals/openpose/build/python/openpose/Release' )
os.environ[ 'PATH' ] = os.environ[ 'PATH' ] + ';' + dir_path + '/../../externals/openpose/build/x64/Release;' + dir_path + '/../../externals/openpose/build/bin;'
import pyopenpose as op


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "video_path", type=str, help="Path to file you want to estimate" )
    parser.add_argument( "-m", "--model", default="michalek", help="Path to model relative to /data/models." )
    parser.add_argument( "-d", "--depth", help="Depth frames will be used to estimation as well", action="store_true" )
    parser.add_argument( "-v", "--view", help="View only mode", action="store_true" )
    parser.add_argument( "-p", "--proceed", help="Pose will be estimated", action="store_true" )
    return parser.parse_known_args()


def getOpenPoseArgs( params ):
    for param in range( 0, len( allArgs[ 1 ] ) ):
        curr_item = allArgs[ 1 ][ param ]
        if param != len( allArgs[ 1 ] ) - 1:
            next_item = allArgs[ 1 ][ param + 1 ]
        else:
            next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace( '-', '' )
            if key not in params:
                params[ key ] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace( '-', '' )
            if key not in params:
                params[ key ] = next_item


def initOpenPose():
    # starting OpenPose
    params = dict()
    params[ "model_folder" ] = "../../externals/openpose/models/"
    getOpenPoseArgs( params )
    wrapper = op.WrapperPython()
    wrapper.configure( params )
    wrapper.start()
    return wrapper


def initFrameDimensions():
    c.frameHeight, c.frameWidth, dim = frameRGB.shape
    c.depthHeight, c.depthWidth = frameD.shape


def getModel():
    mod = tf.keras.models.load_model( '../../data/models/' + args.model )
    print( "Model " + args.model + " loaded." )
    # print( mod.summary() )
    return mod


def getColorFrame():
    if oni:
        frameColor = colorStream.read_frame()
        frameColor = np.array( ( frameColor.height, frameColor.width, 3 ), dtype=np.uint8,
                               buffer=frameColor.get_buffer_as_uint8() ) / 255
    else:
        ret, frameColor = vid.read()
    return frameColor


def getDepthFrame():
    if not args.depth:
        frameDepth = np.zeros( ( c.depthHeight, c.depthWidth ) )
    else:
        frameDepth = depthStream.read_frame()
        frameDepth = np.array( ( frameDepth.height, frameDepth.width ), dtype=np.uint16,
                               buffer=frameDepth.get_buffer_as_uint16() )
    return frameDepth


def proceedFrame():
    datum = op.Datum()
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop( [ datum ] )

    # array of people with keypoints is in datum.poseKeypoints
    # getSkeletons gives for every human skeleton image
    image = datum.cvOutputData  # image is frame with drawn skeleton

    # convert frame to skeleton image
    pos = []
    if args.proceed:
        pos = frame.proceedFrame( mapToRGBD( datum.poseKeypoints, frameD ) )

    return image, pos, datum.poseKeypoints


if __name__ == '__main__':
    allArgs = parseArgs()
    args = allArgs[ 0 ]
    dataPath = None

    if not os.path.isfile( args.video_path ):
        args.video_path = "../../data/videos/" + args.video_path
        if not os.path.isfile( args.video_path ):
            print( "No video found. Please make sure you typed correct path to your video." )
            print( "No video found. Please make sure you typed correct path to your video." )
            exit()

    # OpenNI file
    if args.video_path[ -4: ] == ".oni":
        oni = True
        print( "OpenNI file" )
        vid = openni2.Device.open_file( args.video_path )
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
        vid = cv2.VideoCapture( args.video_path )
        framesNumber = int( vid.get( cv2.CAP_PROP_FRAME_COUNT ) )

    # initialise openPose and Frame class
    if not args.view:
        opWrapper = initOpenPose()
        frame = Frame( getModel() )

    t = time()
    # main loop
    for i in range( framesNumber ):
        frameRGB = getColorFrame()
        frameD = getDepthFrame()
        if i is 0:
            initFrameDimensions()

        if not args.view:
            frameRGB, poses, humans = proceedFrame()
            for j, human in enumerate( humans ):
                print( "skeleton: " + str( poses[ j ][ 1 ] ) + "\tposes: " + str( poses[ j ][ 0 ] ) )
                display.displayPose( frameRGB, human, str( poses[ j ][ 1 ] ) + ": " + c.poses[ np.argmax( poses[ j ][ 0 ] ) ] )

        display.displayFrameTime( frameRGB, time() - t )
        t = time()
        cv2.imshow( "Video frame", frameRGB )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break
