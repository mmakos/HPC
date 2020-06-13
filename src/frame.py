from math import sqrt
import src.consts as c
from src.skeleton import Skeleton


# class to proceed frames, remembering previous skeletons ec.
class Frame:
    def __init__( self ):
        self.skeletons = []

    # humans is list of humans with list of keypoints for every human
    def proceedFrame( self, humans ):
        newSkeletons = []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        self.skeletons = newSkeletons

    def proceedHuman( self, human, newSkeletons ):
        sameSkeletonProb = []            # probability, that human is 'i' skeleton
        minDelta = getMinDelta( getBoundingBox( human ) )
        for i, skeleton in enumerate( self.skeletons ):
            sameSkeletonProb[ i ] = skeleton.compareSkeleton( human, minDelta )
        maxProb = max( sameSkeletonProb )
        if maxProb >= c.probThreshold:      # skeletons are the same human
            i = sameSkeletonProb.index( maxProb )
            self.skeletons[ i ].updateSkeleton( human )     # update skeleton
            newSkeletons.append( self.skeletons[ i ] )      # add skeleton to new skeletons
            self.skeletons.pop( i )                         # skeleton cannot be compared again
        else:
            newSkeletons.append( Skeleton( human ) )        # make new skeleton if there is no similar skeleton



# returns tuple ( width, height, depth )
def getBoundingBox( keypoints ):
    return ( max( keypoints[ : ][ 0 ] ) - min( keypoints[ : ][ 0 ] ),
             max( keypoints[ : ][ 1 ] ) - min( keypoints[ : ][ 1 ] ),
             max( keypoints[ : ][ 2 ] ) - min( keypoints[ : ][ 2 ] ) )


def getMinDelta( boundingBox ):
    return c.minDeltaCoefficient * sqrt( pow( boundingBox[ 0 ], 2 ) +
                                         pow( boundingBox[ 1 ], 2 ) +
                                         pow( boundingBox[ 2 ], 2 ) )