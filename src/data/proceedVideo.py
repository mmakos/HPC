import argparse
import numpy as np
import cv2
import sys
import os
from time import time

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
    parser.add_argument( "video_path", type=str, help="Path to file you want to proceed relative to /data." )
    parser.add_argument( "-v", "--view", help="View only mode.", action="store_true" )
    parser.add_argument( "-p", "--proceed", help="Frames will be converted to skeleton image and saved in given path." )
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


def getStreams():
    global vType, colorStream, depthStream, vid, framesNumber
    if args.video_path[ -4: ] == ".oni":
        vType = 'oni'
        print( "OpenNI file" )
        from primesense import openni2
        vid = openni2.Device.open_file( args.video_path )
        colorStream = vid.create_color_stream()
        colorStream.start()
        depthStream = vid.create_depth_stream()
        depthStream.start()
        framesNumber = colorStream.get_number_of_frames()
    # RealSense video / tiago video
    elif args.video_path[ -4: ] == ".bag":
        vType = 'bag'
        print( "RealSense file" )
        import pyrealsense2 as rs
        vid = rs.pipeline()
        conf = rs.config()
        rs.config.enable_device_from_file( conf, args.video_path, repeat_playback=False )
        conf.enable_stream( rs.stream.depth )
        conf.enable_stream( rs.stream.color )
        profile = vid.start( conf )
        playback = profile.get_device().as_playback()
        playback.set_real_time( False )
        framesNumber = "Unknown quantity of"
    # RGB and depth image video (net datasets and tiago recorded to images)
    elif args.video_path[ -1 ] == "/" or args.video_path[ -1 ] == "\\":
        vType = 'img'
        print( "Video from images." )
        vid = sorted( os.listdir( args.video_path ) )  # vid is array of file names
        framesNumber = len( vid )
        colorStream = 0
        depthStream = 1
    # Regular video
    else:
        vType = 'reg'
        print( "Regular video" )
        vid = cv2.VideoCapture( args.video_path )
        framesNumber = int( vid.get( cv2.CAP_PROP_FRAME_COUNT ) )


def getFrame():
    global colorStream, depthStream
    if vType == 'bag':
        frames = vid.poll_for_frames()
        frameDepth = np.asanyarray( frames.get_depth_frame().get_data() )
        frameColor = cv2.cvtColor( np.asanyarray( frames.get_color_frame().get_data() ), cv2.COLOR_BGR2RGB )
    elif vType == 'oni':
        frameColor = colorStream.read_frame()
        frameColor = np.array( ( frameColor.height, frameColor.width, 3 ), dtype=np.uint8,
                               buffer=frameColor.get_buffer_as_uint8() ) / 255
        frameDepth = depthStream.read_frame()
        frameDepth = np.array( (frameDepth.height, frameDepth.width), dtype=np.uint16,
                               buffer=frameDepth.get_buffer_as_uint16() )
    elif vType == 'img':
        try:
            frameColor = cv2.imread( args.video_path + vid[ colorStream ] )
            frameDepth = cv2.imread( args.video_path + vid[ depthStream ], cv2.IMREAD_ANYDEPTH )
            colorStream = colorStream + 2
            depthStream = depthStream + 2
        except:
            raise FileNotFoundError( "No more frames." )
    elif vType == 'reg':
        frameDepth = np.zeros( ( c.depthHeight, c.depthWidth ) )
        ret, frameColor = vid.read()
    else:
        raise TypeError( "Unsupported video type.")
    return frameColor, frameDepth


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

    return image, skeletons, datum.poseKeypoints


if __name__ == '__main__':
    allArgs = parseArgs()
    args = allArgs[ 0 ]
    dataPath = None
    if args.proceed:
        dataPath = f"../../data/images/{ args.proceed }"
        try:
            os.mkdir( dataPath )
        except:
            pass

    if not os.path.isfile( args.video_path ) and not os.path.isdir( args.video_path ):
        args.video_path = "../../data/" + args.video_path
        if not os.path.isfile( args.video_path ) and not os.path.isdir( args.video_path ):
            print( "No video found. Please make sure you typed correct path to your video." )
            exit()

    global vType, colorStream, depthStream, vid, framesNumber
    getStreams()
    print( framesNumber, "frames to proceed." )
    # initialise openPose and Frame class
    if not args.view:
        opWrapper = initOpenPose()
        frame = Frame()

    t = time()
    savedImgNumber = 0
    i = 0
    # main loop
    for i in range( sys.maxsize ):
        try:
            frameRGB, frameD = getFrame()
        except:
            break
        if i is 0:
            initFrameDimensions()

        if not args.view:
            frameRGB, skeletonImages, human = proceedFrame()
            try:
                for j, img in enumerate( skeletonImages ):
                    cv2.imwrite( f"{ dataPath }/f{ i }s{ img[ 1 ] }.png", 255 * cv2.rotate( img[ 0 ], cv2.ROTATE_90_CLOCKWISE ) )
                    display.displayPose( frameRGB, human[ j ], str( img[ 1 ] ) )
                    savedImgNumber = savedImgNumber + 1
            except:
                break

        display.displayFrameTime( frameRGB, time() - t )
        t = time()
        cv2.imshow( "Video frame", frameRGB )
        cv2.imshow( "Depth frame", frameD )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):

            break

    print( "Finished.\nWritten", savedImgNumber, "files." )
    print( "Proceeded", i, "frames." )
