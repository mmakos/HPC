import numpy as np

name = "4poses.npz"
outName = "4poses_noz.npz"

with np.load( "../../data/datasets/" + name, allow_pickle=True ) as data:
    images = data[ 'images' ]
    labels = data[ 'labels' ]
print( "Dataset loaded." )

for i, img in enumerate( images ):
    images[ i, :, :, 0 ] = np.zeros( [ images[ i ].shape[ 0 ], images[ i ].shape[ 1 ] ] )
print( "converted" )

np.savez_compressed( "../../data/datasets/" + outName, images=images, labels=labels )
print( "saved" )