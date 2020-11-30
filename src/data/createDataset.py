import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from sklearn.utils import shuffle
import sys
sys.path.insert( 1, '../func' )
import consts as c

parser = argparse.ArgumentParser()
parser.add_argument( "poses", help="Path to folder with your poses folders relative to /data/images." )
parser.add_argument( "-d", "--dataset_name", help="Path to output dataset file relative to /data/datasets." )
args = parser.parse_known_args()[ 0 ]

path = "../../data/images/" + args.poses
if args.dataset_name is None:
    args.dataset_name = args.poses
dsPath = "../../data/datasets/" + args.dataset_name

datasetImages = []
datasetLabels = []
label = 0
for _, poses, _ in os.walk( path ):
    for pose in poses:
        posePath = path + "/" + pose
        label = int( pose.split( "_" )[ 1 ] )
        for _, _, images in os.walk( posePath ):
            for i in tqdm( range( len( images ) ), desc=c.poses[ label ] ):
                imgPath = posePath + "/" + images[ i ]
                img = cv2.imread( imgPath )
                datasetImages.append( img )
                datasetLabels.append( label )

datasetImages = np.array( datasetImages )
datasetLabels = np.array( datasetLabels )
datasetImages, datasetLabels = shuffle( datasetImages, datasetLabels )
np.savez_compressed( dsPath, images=datasetImages, labels=datasetLabels )
print( "\nDataset created and saved to file /data/datasets/" + args.dataset_name + ".npz" )



