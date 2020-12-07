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

parser = argparse.ArgumentParser()
parser.add_argument( "input", help="Path to folder with original images relative to /data/images." )
parser.add_argument( "-o", "--output", help="Path to folder with augmented images relative to /data/images." )
args = parser.parse_known_args()[ 0 ]

inputPath = "../../data/images/" + args.input + "/"
if not args.output:
    args.output = args.input + "_aug"
outputPath = "../../data/images/" + args.output + "/"
if not os.path.isdir( outputPath ):
    os.mkdir( outputPath )

images = natsorted( os.listdir( inputPath ) )
print( images )
# creating small pics for different fps
for i in tqdm( range( len( images ) ), desc="Proceed" ):
    if images[ i ][ -4:] != '.png':
        continue
    skeleton = cv2.imread( inputPath + images[ i ] )
    skeletonLength = skeleton.shape[ 1 ]
    for fps in range( c.minOutputFrameRate, c.maxOutputFrameRate + 1, c.frameRateStep ):
        image = []
        length = int( c.imgFrameRate * c.framesNumber / fps )  # length of subimage
        if length > skeletonLength:
            continue
        for start in range( skeletonLength - length + 1 ):
            image = cv2.resize( skeleton[ :, start: start + length ], ( c.framesNumber, c.keypointsNumber ), interpolation=cv2.INTER_AREA )
            cv2.imwrite( f"{ outputPath }{ images[ i ].split( '.' )[ 0 ] }fps{ fps }start{ start }.png", image )
