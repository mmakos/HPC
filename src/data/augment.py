import cv2
import sys
from time import sleep

sys.path.insert( 1, '../func' )
import consts as c
import os
import argparse
from tqdm import tqdm
from natsort import natsorted
import numpy as np
from math import cos, sin, radians


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "input", help="Path to folder with original images relative to /data/images." )
    parser.add_argument( "-o", "--output", help="Path to folder with augmented images relative to /data/images." )
    parser.add_argument( "-s", "--small", help="Path to the folder with small images. If none, then small images won't be created." )
    parser.add_argument( "-m", "--mirror", help="Select if you want to mirror poses.", action="store_true" )
    parser.add_argument( "-r", "--rotate", help="Select if you want to rotate poses.", action="store_true" )
    return parser.parse_known_args()[ 0 ]


def mirrorImage( im ):
    for pair in ( ( 2, 5 ), (  3, 6 ), ( 4, 7 ), ( 9, 12 ), ( 10, 13 ), ( 11, 14 ) ):
        im[ [ pair[ 0 ], pair[ 1 ] ] ] = im[ [ pair[ 1 ], pair[ 0 ] ] ]
    return im


def rotate180( im ):
    for kp, _ in enumerate( im ):
        im[ kp, :, 0 ], im[ kp, :, 2 ] = 255 - im[ kp, :, 0 ], 255 - im[ kp, :, 2 ]
    return im


def rotate( im, angle ):
    mat = np.array( [
        [ cos( angle ), 0, sin( angle ) ],
        [ 0, 1, 0 ],
        [ -sin( angle ), 0, cos( angle ) ]
    ] )
    for kp, _ in enumerate( im ):
        for f, _ in enumerate( im[ kp ] ):
            im[ kp, f ] = mat @ im[ kp, f ]
    # normalize
    mins = ( np.min( im[ :, :, 0 ] ), np.min( im[ :, :, 1 ] ), np.min( im[ :, :, 2 ] ) )
    maxes = ( np.max( im[ :, :, 0 ] ), np.max( im[ :, :, 1 ] ), np.max( im[ :, :, 2 ] ) )
    bbDims = ( maxes[ 0 ] - mins[ 0 ],
               maxes[ 1 ] - mins[ 1 ],
               maxes[ 2 ] - mins[ 2 ] )
    for kp, _ in enumerate( im ):
        for f, _ in enumerate( im[ kp ] ):
            im[ kp, f ] = np.array( [
                ( im[ kp, f, 0 ] - mins[ 0 ] ) * 255 / bbDims[ 0 ],
                ( im[ kp, f, 1 ] - mins[ 1 ] ) * 255 / bbDims[ 1 ],
                ( im[ kp, f, 2 ] - mins[ 2 ] ) * 255 / bbDims[ 2 ] ] )
    return im


if __name__ == "__main__":
    args = parseArgs()
    inputPath = "../../data/images/" + args.input + "/"
    if not args.output:
        args.output = args.input + "_aug"
    outputPath = "../../data/images/" + args.output + "/"
    if not os.path.isdir( outputPath ) and ( args.mirror or args.rotate ):
        os.mkdir( outputPath )
    if args.small is not None:
        smallPath = "../../data/images/" + args.small + "/"
        if not os.path.isdir( smallPath ):
            os.mkdir( smallPath )

    # read images to proceed
    images = natsorted( [ i for i in os.listdir( inputPath ) if i[ -4: ] == '.png' ] )
    print( images )

    if args.mirror:
        for img in images:
            cv2.imwrite( outputPath + img[ :-4 ] + '_m.png', mirrorImage( cv2.imread( inputPath + img ) ) )

    if args.rotate:
        for ang in tqdm( range( 1, 360 ), desc="Rotate" ):
            for img in images:
                cv2.imwrite( f"{ outputPath }{ img[ :-4 ] }_r{ ang }.png", rotate( cv2.imread( inputPath + img ).astype( 'float' ), radians( ang ) ) )

    if args.small is not None:
        # creating small pics for different fps
        for i in tqdm( range( len( images ) ), desc="Make small" ):
            skeleton = cv2.imread( inputPath + images[ i ] )
            skeletonLength = skeleton.shape[ 1 ]
            for fps in range( c.minOutputFrameRate, c.maxOutputFrameRate + 1, c.frameRateStep ):
                image = []
                length = int( c.imgFrameRate * c.framesNumber / fps )  # length of subimage
                if length > skeletonLength:
                    continue
                for start in range( skeletonLength - length + 1 ):
                    image = cv2.resize( skeleton[ :, start: start + length ], ( c.framesNumber, c.keypointsNumber ),
                                        interpolation=cv2.INTER_AREA )
                    cv2.imwrite( f"{ smallPath }{ images[ i ].split( '.' )[ 0 ] }fps{ fps }start{ start }.png", image )
