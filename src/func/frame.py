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
        self.frameTime = 0

    # function returns probabilities list of each pose for each given human (with skeleton id)
    # humans is list of humans with list of keypoints for every human and probability of this keypoint (0, when it's lower than threshold)
    def proceedFrame( self, humans ):
        if self.live:
            self.frameTime = min( self.prevTime - time(), c.maxFrameTime )
            self.prevTime = time()
        newSkeletons = []       # here will be all detected skeletons
        if not humans:
            return []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        # self.skeletons = [ s for s in newSkeletons if s is not None ]   # we save only existing skeletons
        self.skeletons = newSkeletons
        poses = []
        for skeleton in newSkeletons:
            # if skeleton is not None:
            poses.append( [ self.classifyPose( skeleton ), skeleton.getSkeletonId() ] )
            # else:
            #     poses.append( None )
        return poses

    def proceedHuman( self, human, newSkeletons ):
        sameSkeletonProb = []            # probability, that human is 'i' skeleton
        bb = getBoundingBox( human )
        minDelta = getMinDelta( bb )
        for skeleton in self.skeletons:
            sameSkeletonProb.append( skeleton.compareSkeleton( human, minDelta ) )
        if len( sameSkeletonProb ) != 0:
            maxProb = max( sameSkeletonProb )
        else:
            maxProb = 0
        if maxProb >= c.probThreshold:      # skeletons are the same human
            i = sameSkeletonProb.index( maxProb )
            self.skeletons[ i ].updateSkeleton( human, bb )     # update skeleton
            newSkeletons.append( self.skeletons[ i ] )      # add skeleton to new skeletons
            self.skeletons.pop( i )                         # skeleton cannot be compared again
        else:
            newSkeletons.append( Skeleton( human, self.lastSkeletonId, bb ) )        # make new skeleton if there is no similar skeleton
            self.lastSkeletonId = self.lastSkeletonId + 1

    # functions classify pose and returns probabilities of poses
    def classifyPose( self, skeleton ):
        return self.model.predict( ( cv2.rotate( skeleton.getSkeletonImg(), cv2.ROTATE_90_CLOCKWISE ) * 255 ).reshape( -1, c.keypointsNumber, c.framesNumber, 3 ) )

    # function takes detected humans keypoints and return skeleton image for each human
    # this is equivalent to proceedFrame, but for creating dataset
    def getSkeletons( self, humans ):
        newSkeletons = []
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        # self.skeletons = [ s for s in newSkeletons if s is not None ]
        self.skeletons = newSkeletons
        images = []
        for skeleton in self.skeletons:
            # if skeleton is not None:
            images.append( [ skeleton.getSkeletonImg(), skeleton.getSkeletonId() ] )
            # else:
            #     images.append( None )
        return images

    # function does the same as getSkeletons() but it returns last skeleton keypoints instead of image
    def getKeypoints( self, humans ):
        newSkeletons = [ ]
        for human in humans:
            self.proceedHuman( human, newSkeletons )
        self.skeletons = newSkeletons
        keypoints = []
        for skeleton in self.skeletons:
            keypoints.append( [ skeleton.getSkeletonKeypoints(), skeleton.getSkeletonId() ] )
        return keypoints


# returns list [ [ maxW, minW ], [ maxH, minH ], [ maxD, minD ] ]
def getBoundingBox( keypoints ):
    maxmins = [ [ 0, c.frameWidth ], [ 0, c.frameHeight ], [ 0, c.frameDepth ] ]
    for keypoint in keypoints:
        if keypoint[ 3 ] != 0.0:       # if keypoint detected
            for i in range( 3 ):
                if keypoint[ i ] > maxmins[ i ][ 0 ]:
                    maxmins[ i ][ 0 ] = keypoint[ i ]
                if keypoint[ i ] < maxmins[ i ][ 1 ]:
                    maxmins[ i ][ 1 ] = keypoint[ i ]
    return maxmins


def getMinDelta( boundingBox ):
    return c.maxDeltaCoefficient * sqrt( pow( boundingBox[ 0 ][ 0 ] - boundingBox[ 0 ][ 1 ], 2 ) +
                                         pow( boundingBox[ 1 ][ 0 ] - boundingBox[ 1 ][ 1 ], 2 ) )
