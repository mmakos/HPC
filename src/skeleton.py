# Class which contains single skeletons information
# and can code it to image

import numpy as np
import src.consts as c
from math import sqrt


class Skeleton:
    # last keypoints are keypoints of skeleton from previous frame
    # to create new skeleton we have to give actual keypoints of this skeleton
    # maxDimensions are used to reduce coordinates to [0, 1]
    def __init__( self, keypoints ):
        self.lastKeypoints = keypoints
        self.skeletonImg = np.zeros( ( c.keypointsNumber, c.framesNumber, 3 ) )

    # Functions updates skeleton from given frame keypoints (original coordinates)
    def updateSkeleton( self, keypoints ):
        self.lastKeypoints = keypoints
        self.updateImg()

    def updateImg( self ):
        # all columns (frames) need to be swap left
        for i in range( c.framesNumber - 1 ):
            self.skeletonImg[ i ] = self.skeletonImg[ i + 1 ]
        self.skeletonImg[ c.framesNumber - 1 ] = self.lastKeypoints / 255

    # function returns probability, that skeleton a i b is the same skeleton
    # keypoints - skeleton A
    # self.lastKeypoints - skeleton B
    # minDelta - is computed only once for skeleton a
    def compareSkeleton( self, keypoints, minDelta ):
        sab = []  # Sab - probabilities that point i of a and b is from the same skeleton
        for i, point in enumerate( keypoints ):
            sab[ i ] = 1 - ( sqrt( pow( point[ 0 ] - self.lastKeypoints[ i ][ 0 ], 2 ) +
                                   pow( point[ 1 ] - self.lastKeypoints[ i ][ 1 ], 2 ) +
                                   pow( point[ 2 ] - self.lastKeypoints[ i ][ 2 ], 2 ) ) / minDelta )
            if sab[ i ] < 0:
                sab[ i ] = 0
        return np.mean( sab )

    def getSkeletonImg( self ):
        return self.skeletonImg
