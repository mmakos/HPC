import numpy as np
from tqdm import tqdm

name = "jumpwalkVal.npz"
outName = "jumpwalkVal_noz.npz"

with np.load( "../../data/datasets/" + name, allow_pickle=True ) as data:
    images = data[ 'images' ]
    labels = data[ 'labels' ]
print( "Dataset loaded." )

for i in tqdm( range( len( images ) ), desc="Images" ):
    images[ i, :, :, 0 ] = np.zeros( [ images[ i ].shape[ 0 ], images[ i ].shape[ 1 ] ] )
print( "converted" )

np.savez_compressed( "../../data/datasets/" + outName, images=images, labels=labels )
print( "saved" )

with np.load( "../../data/datasets/" + outName, allow_pickle=True ) as data:
    images = data[ 'images' ]
    labels = data[ 'labels' ]
