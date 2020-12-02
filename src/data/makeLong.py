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
parser.add_argument( "-o", "--output", help="Path to folder where you want to save whole skeleton images (long images)." )
parser.add_argument( "-z", "--no_z", help="Z dimension will not be proceeded.", action="store_true" )
args = parser.parse_known_args()[ 0 ]

inputPath = "../../data/images/" + args.input + "/"

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

if args.output is None:
    args.output = args.input + "_long"
savePath = "../../data/images/" + args.output + "/"
if not os.path.isdir( savePath ):
    os.mkdir( savePath )
for i, img in enumerate( skelImgs ):
    im = np.concatenate( img, axis=1 )
    if args.no_z:
        for x, _ in enumerate( im ):
            for y, _ in enumerate( im[ x ] ):
                im[ x ][ y ][ 2 ] = 0
    cv2.imwrite( f"{ savePath }s{ skels[ i ] }.png", im )
