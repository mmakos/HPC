# Class which contains single skeletons information
# and can code it to image

import numpy as np
import consts as c
from math import sqrt


class Skeleton:
    # last keypoints are keypoints of skeleton from previous frame
    # to create new skeleton we have to give actual keypoints of this skeleton
    # coordinates are normalised to [0, 1]
    def __init__( self, keypoints, skeletonId, boundingBox ):
        self.lastKeypoints = keypoints
        self.id = skeletonId
        self.boundingBox = boundingBox  # bounding box of last keypoints set (for faster operations)
        self.skeletonImg = np.zeros( ( c.framesNumber, c.keypointsNumber, 3 ) )
        self.skeletonImg[ c.framesNumber - 1 ] = normalize( keypoints, boundingBox )

    # Functions updates skeleton from given frame keypoints (original coordinates)
    def updateSkeleton( self, keypoints, boundingBox ):
        self.lastKeypoints = keypoints
        self.boundingBox = boundingBox
        self.updateImg()

    def updateImg( self ):
        # all columns (frames) need to be swap left
        for i in range( c.framesNumber - 1 ):
            self.skeletonImg[ i ] = self.skeletonImg[ i + 1 ]
        self.skeletonImg[ c.framesNumber - 1 ] = normalize( self.lastKeypoints, self.boundingBox )   # normalization

    # function returns probability, that skeleton a i b is the same skeleton
    # keypoints - skeleton A
    # self.lastKeypoints - skeleton B
    # minDelta - is computed only once for skeleton a
    def compareSkeleton( self, keypoints, minDelta ):
        sab = []  # Sab - probabilities that point i of a and b is from the same skeleton
        for i, point in enumerate( keypoints ):
            if point[ 3 ] == 0.0 or self.lastKeypoints[ i ][ 3 ] == 0.0:     # we count only if points exists
                sab.append( 0 )
                continue
            sab.append( 1 - ( sqrt( pow( int( point[ 0 ] - self.lastKeypoints[ i ][ 0 ] ), 2 ) +
                                    pow( int( point[ 1 ] - self.lastKeypoints[ i ][ 1 ] ), 2 ) ) / minDelta ) )
            if sab[ i ] < 0:
                sab[ i ] = 0
        return np.mean( sab )

    def getSkeletonImg( self ):
        return self.skeletonImg

    def getSkeletonId( self ):
        return self.id


def normalize( keypoints, boundingBox ):
    bbDims = [ boundingBox[ 0 ][ 0 ] - boundingBox[ 0 ][ 1 ],
               boundingBox[ 1 ][ 0 ] - boundingBox[ 1 ][ 1 ],
               boundingBox[ 2 ][ 0 ] - boundingBox[ 2 ][ 1 ] ]

    try:
        return [ [ ( i[ 0 ] - boundingBox[ 0 ][ 1 ] ) / bbDims[ 0 ],
                   ( i[ 1 ] - boundingBox[ 1 ][ 1 ] ) / bbDims[ 1 ],
                   ( i[ 2 ] - boundingBox[ 2 ][ 1 ] ) / bbDims[ 2 ] ]
                 for i in keypoints ]
    except ZeroDivisionError:
        return [ [ ( i[ 0 ] - boundingBox[ 0 ][ 1 ] ) / ( bbDims[ 0 ] ),
                   ( i[ 1 ] - boundingBox[ 1 ][ 1 ] ) / ( bbDims[ 1 ] ),
                   ( i[ 2 ] - boundingBox[ 2 ][ 1 ] ) / ( bbDims[ 2 ] + 1 ) ]
                 for i in keypoints ]
