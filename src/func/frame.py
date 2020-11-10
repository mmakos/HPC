from math import sqrt
import consts as c
from skeleton import Skeleton
import cv2


# class to proceed frames, remembering previous skeletons ec.
class Frame:
    def __init__( self, model=None ):
        self.skeletons = []
        self.model = model
        self.maxSkeletonId = 0

    # function returns probabilities list of each pose for each given human
    # humans is list of humans with list of keypoints for every human
    def proceedFrame( self, humans ):
        newSkeletons = []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        self.skeletons = newSkeletons
        poses = []
        for skeleton in self.skeletons:
            poses.append( [ self.classifyPose( skeleton ), skeleton.getSkeletonId() ] )
        return poses

    def proceedHuman( self, human, newSkeletons ):
        sameSkeletonProb = []            # probability, that human is 'i' skeleton
        minDelta = getMinDelta( getBoundingBox( human ) )
        for skeleton in self.skeletons:
            sameSkeletonProb.append( skeleton.compareSkeleton( human, minDelta ) )
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
            newSkeletons.append( Skeleton( human, self.maxSkeletonId ) )        # make new skeleton if there is no similar skeleton
            self.maxSkeletonId = self.maxSkeletonId + 1

    # functions classify pose and returns probabilities of poses
    def classifyPose( self, skeleton ):
        return self.model.predict( ( cv2.rotate( skeleton.getSkeletonImg(), cv2.ROTATE_90_CLOCKWISE ) * 255 ).reshape( -1, c.keypointsNumber, c.framesNumber, 3 ) )
        # return [ 1., 0., 0., 0., 0. ]

    # function takes detected humans keypoints and return skeleton image for each human
    # this is equivalent to proceedFrame, but for creating dataset
    def getSkeletons( self, humans ):
        newSkeletons = []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        self.skeletons = newSkeletons
        return [ [ skeleton.getSkeletonImg(), skeleton.getSkeletonId() ] for skeleton in self.skeletons ]


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
