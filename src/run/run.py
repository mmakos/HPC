import argparse
from camera import Camera
from wrapper import Wrapper
import cv2
import sys
sys.path.insert( 1, '../func' )
import consts as c
import os


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "-v", "--video", type=str, help="Name of video you want to estimate relative to /data/videos." )
    parser.add_argument( "-c", "--hybrid", help="Select if you want to use hybrid method.", action="store_true" )
    parser.add_argument( "-w", "--write_name", help="Name of output video. If none, video will not be saved." )
    parser.add_argument( "-p", "--no_pose", help="Pose will be estimated.", action="store_true" )
    parser.add_argument( "-g", "--gpu_mode", help="Pose classification will be executed on GPU, but GPU can be out of memory", action="store_true" )
    parser.add_argument( "-d", "--no_depth", help="Depth canal will be excluded.", action="store_true" )
    return parser.parse_known_args()


def getOpParams():
    params = dict()
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
    return params


if __name__ == "__main__":
    allArgs = parseArgs()
    args = allArgs[ 0 ]
    dataPath = out = None

    # change video path to relative to /data/videos
    if args.video is not None:
        args.video = "../../data/videos/" + args.video
        if not os.path.isfile( args.video ) and not os.path.isdir( args.video ):
            print( "No video found. Please make sure you typed correct path to your video." )
            exit()
    # change output video name to relative (if needed)
    if args.write_name is not None:
        args.write_name = "../../data/videos/" + args.write_name + ".mp4"

    # create camera stream or video
    camera = Camera( video=args.video, noDepth=args.no_depth )
    # create wrapper (it will do all to classify poses on image)
    wrapper = Wrapper( model=( "static", "dynamic" ) if args.hybrid else "allRot",
                       gpuMode=args.gpu_mode, opParams=getOpParams() )

    firstFrame = True
    while True:
        try:
            # THIS WILL RETURN:
            # image with drawn skeletons, bounding boxes and written poses
            # list of classified poses with human unique index
            # list of humans in the same order as poses
            # camera.getFrame() returns ( frameRGB, frameDepth ) from available stream
            img, _, _ = wrapper.proceed( camera.getFrame(), noDepth=args.no_depth, noPose=args.no_pose )
            # show image
            cv2.imshow( "Video", img )

            # if we want to create output video (now frame dimensions are initialized)
            if firstFrame and args.write_name is not None:
                firstFrame = False
                out = cv2.VideoWriter( args.write_name, cv2.VideoWriter_fourcc( *'mp4v' ), 30,
                                       (c.frameWidth, c.frameHeight) )
            # write image to output video
            if out is not None:
                out.write( img )
        # camera.getFrame() will cause EOFError if video has ended
        except EOFError:
            print( "End of video." )
            break

        # press q to quit program
        if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
            break

    # save video after quiting program or end of video
    if out is not None:
        out.release()
