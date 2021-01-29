# Class which contains single skeletons information
# and can code it to image

import numpy as np
import consts as c
from math import sqrt
from preprocess import estimateNotDetectedKeypoints


class Skeleton:
    # last keypoints are keypoints of skeleton from previous frame
    # to create new skeleton we have to give actual keypoints of this skeleton
    # coordinates are normalised to [0, 1]
    def __init__( self, keypoints, skeletonId, boundingBox ):
        estimateNotDetectedKeypoints( keypoints )
        self.lastKeypoints = keypoints
        self.id = skeletonId
        self.boundingBox = boundingBox  # bounding box of last keypoints set (for faster operations)
        self.skeletonImg = np.zeros( ( c.framesNumber, c.keypointsNumber, 3 ) )
        normalisedKeypoints = normalize( keypoints, self.boundingBox )
        for i in range( c.framesNumber ):
            self.skeletonImg[ i ] = normalisedKeypoints
        # self.skeletonImg[ c.framesNumber - 1 ] = normalize( keypoints, boundingBox )

    # Functions updates skeleton from given frame keypoints (original coordinates)
    def updateSkeleton( self, keypoints, boundingBox ):
        # after tracking we can estimate not detected keypoints
        # but only by symmetry etc (no interpolation of neighbour frames)
        if c.fillNotDetected:
            estimateNotDetectedKeypoints( keypoints )
        self.lastKeypoints = keypoints
        self.boundingBox = boundingBox
        self.__updateImg()

    def __updateImg( self ):
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

    # function returns sum of distances of particular point between all frames
    def getPointsDistance( self, points=( 9, 10, 11, 12, 13, 14 ) ):
        moveSum = np.zeros( shape=3 )
        for k in points:
            for f in range( 1, 32 ):
                moveSum = np.add( moveSum, np.fabs( np.subtract( self.skeletonImg[ k, f ], self.skeletonImg[ k, f - 1 ] ) ) )
        return moveSum[ 0 ] * c.xDistCoefficient + moveSum[ 1 ] * c.yDistCoefficient

    def getSkeletonImg( self ):
        return self.skeletonImg

    def getSkeletonId( self ):
        return self.id

    def getSkeletonKeypoints( self ):
        return self.lastKeypoints


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
        return [ [ ( i[ 0 ] - boundingBox[ 0 ][ 1 ] ) / ( bbDims[ 0 ] + 1 ),
                   ( i[ 1 ] - boundingBox[ 1 ][ 1 ] ) / ( bbDims[ 1 ] + 1 ),
                   ( i[ 2 ] - boundingBox[ 2 ][ 1 ] ) / ( bbDims[ 2 ] + 1 ) ]
                 for i in keypoints ]
