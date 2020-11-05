from math import sqrt
import consts as c
from skeleton import Skeleton
from model import getModel


# class to proceed frames, remembering previous skeletons ec.
class Frame:
    def __init__( self ):
        self.skeletons = []
        self.model = getModel()

    # function returns probabilities list of each pose for each given human
    # humans is list of humans with list of keypoints for every human
    def proceedFrame( self, humans ):
        newSkeletons = []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        self.skeletons = newSkeletons
        poses = []
        for i, skeleton in enumerate( self.skeletons ):
            poses.append( self.classifyPose( skeleton ) )
        return poses

    def proceedHuman( self, human, newSkeletons ):
        sameSkeletonProb = []            # probability, that human is 'i' skeleton
        minDelta = getMinDelta( getBoundingBox( human ) )
        for i, skeleton in enumerate( self.skeletons ):
            sameSkeletonProb[ i ] = skeleton.compareSkeleton( human, minDelta )
        if len( sameSkeletonProb ) != 0:
            maxProb = max( sameSkeletonProb )
        else:
            maxProb = 0
        if maxProb >= c.probThreshold:      # skeletons are the same human
            i = sameSkeletonProb.index( maxProb )
            self.skeletons[ i ].updateSkeleton( human )     # update skeleton
            newSkeletons.append( self.skeletons[ i ] )      # add skeleton to new skeletons
            self.skeletons.pop( i )                         # skeleton cannot be compared again
        else:
            newSkeletons.append( Skeleton( human ) )        # make new skeleton if there is no similar skeleton

    # functions classify pose and returns probabilities of poses
    def classifyPose( self, skeleton ):
        # TODO
        # return self.model.predict( skeleton.getSkeletonImg() )
        return [ 1., 0., 0., 0., 0. ]


# returns tuple ( width, height, depth )
def getBoundingBox( keypoints ):
    maxmins = [ [ 0, c.frameWidth ], [ 0, c.frameHeight ], [ 0, c.frameDepth ] ]
    for keypoint in keypoints:
        if keypoint[ 2 ] > 0:       # if keypoint detected
            for i in range( 3 ):
                if keypoint[ i ] > maxmins[ i ][ 0 ]:
                    maxmins[ i ][ 0 ] = keypoint[ i ]
                if keypoint[ i ] < maxmins[ i ][ 1 ]:
                    maxmins[ i ][ 1 ] = keypoint[ i ]
    return ( maxmins[ 0 ][ 0 ] - maxmins[ 0 ][ 1 ],
             maxmins[ 1 ][ 0 ] - maxmins[ 1 ][ 1 ],
             maxmins[ 2 ][ 0 ] - maxmins[ 2 ][ 1 ] )


def getMinDelta( boundingBox ):
    return c.minDeltaCoefficient * sqrt( pow( boundingBox[ 0 ], 2 ) +
                                         pow( boundingBox[ 1 ], 2 ) +
                                         pow( boundingBox[ 2 ], 2 ) )
