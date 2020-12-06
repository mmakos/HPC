import argparse
import numpy as np
import cv2
import sys
import os
from time import time
import pickle

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
    parser.add_argument( "video", type=str, help="Path to file you want to proceed relative to /data/videos." )
    parser.add_argument( "-v", "--view", help="View only mode.", action="store_true" )
    parser.add_argument( "-p", "--proceed", help="Frames will be converted to skeleton image and saved in given path relative to /data/images." )
    parser.add_argument( "-w", "--write_video", help="Video with drawn skeletons and boxes wil be saved with name given into --proceed.", action="store_true" )
    parser.add_argument( "-l", "--long", help="Skeletons will be saved as one long image instead of multiple small images.", action="store_true" )
    parser.add_argument( "-k", "--keypoints_mode", help="Saves pure keypoints into text file in folder given into --proceed.", action="store_true" )
    parser.add_argument( "-a", "--annotations", help="Loads file with skeleton annotations and creates images based on those." )
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
    params[ "render_threshold" ] = c.keypointThreshold
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
    if args.video[ -4: ] == ".oni":
        vType = 'oni'
        print( "OpenNI file" )
        from primesense import openni2
        vid = openni2.Device.open_file( args.video )
        colorStream = vid.create_color_stream()
        colorStream.start()
        depthStream = vid.create_depth_stream()
        depthStream.start()
        framesNumber = colorStream.get_number_of_frames()
    # RealSense video / tiago video
    elif args.video[ -4: ] == ".bag":
        vType = 'bag'
        print( "RealSense file" )
        import pyrealsense2 as rs
        vid = rs.pipeline()
        conf = rs.config()
        rs.config.enable_device_from_file( conf, args.video, repeat_playback=False )
        conf.enable_stream( rs.stream.depth )
        conf.enable_stream( rs.stream.color )
        profile = vid.start( conf )
        playback = profile.get_device().as_playback()
        playback.set_real_time( False )
        framesNumber = "Unknown quantity of"
    # RGB and depth image video (net datasets and tiago recorded to images)
    elif args.video[ -1 ] == "/" or args.video[ -1 ] == "\\":
        vType = 'img'
        print( "Video from images." )
        vid = sorted( os.listdir( args.video ) )  # vid is array of file names
        framesNumber = len( vid )
        colorStream = 0
        depthStream = 1
    # Regular video
    else:
        vType = 'reg'
        print( "Regular video" )
        vid = cv2.VideoCapture( args.video )
        framesNumber = int( vid.get( cv2.CAP_PROP_FRAME_COUNT ) )


# throws EOFError when end of video
def getFrame():
    global colorStream, depthStream
    if vType == 'bag':
        try:
            frames = vid.poll_for_frames()
            frameDepth = np.asanyarray( frames.get_depth_frame().get_data() )
            frameColor = cv2.cvtColor( np.asanyarray( frames.get_color_frame().get_data() ), cv2.COLOR_BGR2RGB )
        except RuntimeError:
            raise EOFError( "No more frames" )
    elif vType == 'oni':
        frameColor = colorStream.read_frame()
        frameColor = np.array( ( frameColor.height, frameColor.width, 3 ), dtype=np.uint8,
                               buffer=frameColor.get_buffer_as_uint8() ) / 255
        frameDepth = depthStream.read_frame()
        frameDepth = np.array( (frameDepth.height, frameDepth.width), dtype=np.uint16,
                               buffer=frameDepth.get_buffer_as_uint16() )
    elif vType == 'img':
        try:
            frameColor = cv2.imread( args.video + vid[ colorStream ] )
            frameDepth = cv2.imread( args.video + vid[ depthStream ], cv2.IMREAD_ANYDEPTH )
            colorStream = colorStream + 2
            depthStream = depthStream + 2
        except IndexError:
            raise EOFError( "No more frames." )
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

    # map to RGBD
    rgbdKeypoints = mapToRGBD( datum.poseKeypoints, frameD )
    # convert frame to skeleton image
    skeletons = []
    if args.proceed:
        if not args.keypoints_mode:
            skeletons = frame.getSkeletons( rgbdKeypoints )
        else:
            skeletons = frame.getKeypoints( rgbdKeypoints )

    return image, skeletons, rgbdKeypoints


def proceedSkeletonImages( end=False ):
    global savedImgNumber
    currentSkels = [ ]
    if not end:
        for j, img in enumerate( skeletonImages ):
            img[ 0 ] = 255 * cv2.rotate( img[ 0 ], cv2.ROTATE_90_CLOCKWISE )
            if not args.long:  # save image for every skeleton
                cv2.imwrite( f"{ dataPath }/f{ i }s{ img[ 1 ] }.png", img[ 0 ] )
                savedImgNumber = savedImgNumber + 1
            else:  # remember to write long video when skeleton ends
                currentSkels.append( img[ 1 ] )
                if img[ 1 ] not in skelNumbers:
                    skelNumbers.append( img[ 1 ] )
                    skels.append( [ ] )
                skels[ skelNumbers.index( img[ 1 ] ) ].insert( 0, img[ 0 ][ :, 0:1 ] )
            display.displayPose( frameRGB, human[ j ], str( img[ 1 ] ) )

    if args.long:  # write ended skeletons
        for si, s in enumerate( skelNumbers ):
            if end or s not in currentSkels:  # end of skeleton
                if len( skels[ si ] ) >= c.minLongImageLength:
                    cv2.imwrite( f"{ dataPath }/s{ s }.png", np.concatenate( skels[ si ], axis=1 ) )  # save
                skels.pop( si )  # remove from lists
                skelNumbers.pop( si )


def proceedSkeletonKeypoints( end=False ):
    global savedImgNumber
    currentSkels = []
    if not end:
        for j, keypoints in enumerate( skeletonImages ):
            currentSkels.append( keypoints[ 1 ] )
            if keypoints[ 1 ] not in skelNumbers:
                skelNumbers.append( keypoints[ 1 ] )
                skels.append( [] )
            skels[ skelNumbers.index( keypoints[ 1 ] ) ].append( keypoints[ 0 ] )
            display.displayPose( frameRGB, human[ j ], str( keypoints[ 1 ] ) )

    # write ended skeletons
    for si, s in enumerate( skelNumbers ):
        if end or s not in currentSkels:  # end of skeleton
            if len( skels[ si ] ) >= c.minLongImageLength:  # save, as sXatY where x is skeleton number and at is frame when it starts to show
                pickle.dump( skels[ si ], open( f"{ dataPath }/s{ s }at{ i - len( skels[ si ] ) }.p", "wb" ) )
                # cv2.imwrite( f"{dataPath}/s{s}.png", np.concatenate( skels[ si ], axis=1 ) )  # save
            skels.pop( si )  # remove from lists
            skelNumbers.pop( si )


if __name__ == '__main__':
    allArgs = parseArgs()
    args = allArgs[ 0 ]
    videoWriter = None
    outputVidPath = None
    dataPath = None
    if args.proceed:
        dataPath = f"../../data/images/{ args.proceed }"
        try:
            os.mkdir( dataPath )
        except:
            pass

    args.video = "../../data/videos/" + args.video
    if not os.path.isfile( args.video ) and not os.path.isdir( args.video ):
        print( "No video found. Please make sure you typed correct path to your video." )
        exit()

    global vType, colorStream, depthStream, vid, framesNumber
    getStreams()
    print( framesNumber, "frames to proceed." )
    # initialise openPose and Frame class
    if not args.view:
        opWrapper = initOpenPose()
        frame = Frame()

    skels = []
    skelNumbers = []
    t = time()
    savedImgNumber = 0
    i = 0
    # main loop
    for i in range( sys.maxsize ):
        try:
            frameRGB, frameD = getFrame()
        except EOFError:
            break
        if i == 0:
            initFrameDimensions()
            if args.write_video and dataPath is not None:
                outputVidPath = dataPath + ".mp4"
                videoWriter = cv2.VideoWriter( outputVidPath, cv2.VideoWriter_fourcc( *'mp4v' ), 30.0,
                                               ( c.frameWidth, c.frameHeight ) )

        if not args.view:
            frameRGB, skeletonImages, human = proceedFrame()
            if not args.keypoints_mode:
                proceedSkeletonImages()
            else:
                proceedSkeletonKeypoints()

        display.displayFrameTime( frameRGB, time() - t )
        display.displayFrameNumber( frameRGB, i )
        t = time()
        cv2.imshow( "Video frame", frameRGB )
        cv2.imshow( "Depth frame", frameD )
        if videoWriter is not None:
            videoWriter.write( frameRGB )
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    if videoWriter is not None:
        videoWriter.release()

    if len( skelNumbers ) > 0:
        if not args.keypoints_mode and args.long:
            proceedSkeletonImages( end=True )
        elif args.keypoints_mode:
            proceedSkeletonKeypoints( end=True )

    print( "Written", savedImgNumber, "skeleton images." )
    print( "Proceeded", i, "frames." )
