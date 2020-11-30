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
parser.add_argument( "-s", "--save", help="Path to folder where you want to save whole skeleton images (long images)." )
args = parser.parse_known_args()[ 0 ]

inputPath = "../../data/images/" + args.input + "/"
if not args.output:
    args.output = args.input + "_aug"
outputPath = "../../data/images/" + args.output + "/"
if not os.path.isdir( outputPath ):
    os.mkdir( outputPath )

images = natsorted( os.listdir( inputPath ) )
maxFrame = int( images[ -1 ].split( 's' )[ 0 ].strip( 'f' ) )
skels = []
for i in images:
    skel = int( i.split( 's' )[ 1 ].split( '.' )[ 0 ] )
    if skel not in skels:
        skels.append( skel )
print( "Following skeletons found:", skels )
sleep( .1 )
skelImgs = [ [] for i in skels ]
for f in tqdm( range( maxFrame + 1 ), desc="Creating whole images" ):
    for si, s in enumerate( skels ):
        try:
            skelImgs[ si ].insert( 0, cv2.imread( f"{ inputPath }f{ f }s{ s }.png" )[ :, 0:1 ] )
        except TypeError:
            pass

skeletonImages = []
for img in skelImgs:
    skeletonImages.append( np.concatenate( img, axis=1 ) )

if args.save is not None:
    savePath = "../../data/images/" + args.save + "/"
    if not os.path.isdir( savePath ):
        os.mkdir( savePath )
    for i, img in enumerate( skelImgs ):
        cv2.imwrite( f"{ savePath }s{ skels[ i ] }.png", np.concatenate( img, axis=1 ) )

# creating small pics for different fps
for si, skeleton in enumerate( skeletonImages ):
    skeletonLength = skeleton.shape[ 1 ]
    for fps in tqdm( range( c.minOutputFrameRate, c.maxOutputFrameRate + 1, c.frameRateStep ), desc=f"Proceeding { skels[ si ] } skeleton" ):
        image = []
        length = int( c.imgFrameRate * c.framesNumber / fps )  # length of subimage
        if length > skeletonLength:
            continue
        for start in range( skeletonLength - length + 1 ):
            image = cv2.resize( skeleton[ :, start: start + length ], ( c.framesNumber, c.keypointsNumber ), interpolation=cv2.INTER_AREA )
            cv2.imwrite( f"{ outputPath }skel{ si }fps{ fps }start{ start }.png", image )
