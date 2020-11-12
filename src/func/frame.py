from math import sqrt
import consts as c
from skeleton import Skeleton
import cv2
from time import time


# class to proceed frames, remembering previous skeletons ec.
class Frame:
    def __init__( self, model=None, live=True ):
        self.skeletons = []
        self.model = model
        self.lastSkeletonId = 0
        self.live = live
        self.prevTime = time()

    # function returns probabilities list of each pose for each given human
    # humans is list of humans with list of keypoints for every human
    def proceedFrame( self, humans ):
        if self.live:
            c.frameTime = self.prevTime - time()
            if c.frameTime > c.maxFrameTime:
                c.frameTime = c.maxFrameTime
            self.prevTime = time()
        newSkeletons = []
        for human in humans:
            if sum( 1 for kp in human if kp != [ 0.0, 0.0, 0.0 ] ) >= c.minDetectedKeypoints:
                self.proceedHuman( human, newSkeletons )
            else:
                newSkeletons.append( None )
        self.skeletons = [ s for s in newSkeletons if s is not None ]
        poses = []
        for skeleton in newSkeletons:
            if skeleton is not None:
                poses.append( [ self.classifyPose( skeleton ), skeleton.getSkeletonId() ] )
            else:
                poses.append( None )
        return poses

    def proceedHuman( self, human, newSkeletons ):
        sameSkeletonProb = []            # probability, that human is 'i' skeleton
        minDelta = getMinDelta( getBoundingBox( human ) )
        print( "\nminDelta = " + str( minDelta ) )
        for skeleton in self.skeletons:
            print( "Comparing to skeleton " + str( skeleton.getSkeletonId() ) )
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
            newSkeletons.append( Skeleton( human, self.lastSkeletonId ) )        # make new skeleton if there is no similar skeleton
            self.lastSkeletonId = self.lastSkeletonId + 1

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
        return [ skeleton.getSkeletonImg() for skeleton in self.skeletons ]


# returns tuple ( width, height, depth )
def getBoundingBox( keypoints ):
    maxmins = [ [ 0, c.frameWidth ], [ 0, c.frameHeight ], [ 0, c.frameDepth ] ]
    for keypoint in keypoints:
        if keypoint != [ 0.0, 0.0, 0.0 ]:       # if keypoint detected
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
