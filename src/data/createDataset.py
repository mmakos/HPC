import os
import numpy as np
import cv2
import argparse
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument( "dataset_name", help="Path to output dataset file relative to /data/datasets." )
parser.add_argument( "-p", "--poses", default="/poses", help="Path to folder with your poses folders relative to /data/images." )
args = parser.parse_known_args()[ 0 ]

path = "../../data/images/" + args.poses

datasetImages = []
datasetLabels = []
label = 0
for _, poses, _ in os.walk( path ):
    for pose in poses:
        posePath = path + "/" + pose
        label = pose.split( "_" )[ 1 ]
        for _, _, images in os.walk( posePath ):
            for image in images:
                imgPath = posePath + "/" + image
                print( imgPath + "\tlabel = " + label + "\t- done." )
                img = cv2.imread( imgPath )
                datasetImages.append( img )
                datasetLabels.append( int( label ) )

datasetImages = np.array( datasetImages )
datasetLabels = np.array( datasetLabels )
datasetImages, datasetLabels = shuffle( datasetImages, datasetLabels )
dsPath = "../../data/datasets/" + args.dataset_name
np.savez_compressed( dsPath, images=datasetImages, labels=datasetLabels )
print( "\nDataset created and saved to file /data/datasets/" + args.dataset_name + ".npz" )



