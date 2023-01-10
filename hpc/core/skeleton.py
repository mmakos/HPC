# Class which contains single skeletons information
# and can code it to image

from math import sqrt

import numpy as np

import hpc.consts as c
from hpc.core.preprocess import detected


class Skeleton:
    # last keypoints are keypoints of skeleton from previous frame
    # to create new skeleton we have to give actual keypoints of this skeleton
    # coordinates are normalised to [0, 1]
    def __init__(self, keypoints, skeletonId, boundingBox):
        self.keypointsScore = np.ones((c.framesNumber, c.keypointsNumber))
        for i in range(c.framesNumber):
            for j in range(c.keypointsNumber):
                self.keypointsScore[i, j] = keypoints[j][-1]
        estimateNotDetectedKeypoints(keypoints)
        self.lastKeypoints = keypoints
        self.id = skeletonId
        self.boundingBox = boundingBox  # bounding box of last keypoints set (for faster operations)
        self.skeletonImg = np.zeros((c.framesNumber, c.keypointsNumber, 3))
        normalisedKeypoints = normalize(keypoints, self.boundingBox)
        for i in range(c.framesNumber):
            self.skeletonImg[i] = normalisedKeypoints
        # self.skeletonImg[ c.framesNumber - 1 ] = normalize( keypoints, boundingBox )

    # Functions updates skeleton from given frame keypoints (original coordinates)
    def updateSkeleton(self, keypoints, boundingBox):
        # after tracking we can estimate not detected keypoints by interpolation or not mirroring and parentizing
        if c.interpolateNotDetected:
            self.__interpolateNotDetectedKeypoints()

        if c.fillNotDetected:
            estimateNotDetectedKeypoints(keypoints)
        self.lastKeypoints = keypoints
        self.boundingBox = boundingBox
        self.__updateImg()

    def __interpolateNotDetectedKeypoints(self):
        for kp in range(c.keypointsNumber):
            begin = -1
            ranges = list()
            for i in range(c.framesNumber):
                if self.keypointsScore[i, kp] < c.keypointThreshold:
                    if begin == -1:
                        begin = i
                elif begin >= 0:
                    ranges.append([begin, i])
        pass


    def __updateImg(self):
        # all columns (frames) need to be swap left
        for i in range(c.framesNumber - 1):
            self.skeletonImg[i] = self.skeletonImg[i + 1]
        self.skeletonImg[c.framesNumber - 1] = normalize(self.lastKeypoints, self.boundingBox)  # normalization

    # function returns probability, that skeleton a i b is the same skeleton
    # keypoints - skeleton A
    # self.lastKeypoints - skeleton B
    # minDelta - is computed only once for skeleton a
    def compareSkeleton(self, keypoints, minDelta):
        sab = []  # Sab - probabilities that point i of a and b is from the same skeleton
        for i, point in enumerate(keypoints):
            if point[3] == 0.0 or self.lastKeypoints[i][3] == 0.0:  # we count only if points exists
                sab.append(0)
                continue
            sab.append(1 - (sqrt(pow(int(point[0] - self.lastKeypoints[i][0]), 2) +
                                 pow(int(point[1] - self.lastKeypoints[i][1]), 2)) / minDelta))
            if sab[i] < 0:
                sab[i] = 0
        return np.mean(sab)

    # function returns sum of distances of particular point between all frames
    def getPointsDistance(self, points=(9, 10, 11, 12, 13, 14)):
        moveSum = np.zeros(shape=3)
        for k in points:
            for f in range(1, 32):
                moveSum = np.add(moveSum, np.fabs(np.subtract(self.skeletonImg[k, f], self.skeletonImg[k, f - 1])))
        return moveSum[0] * c.xDistCoefficient + moveSum[1] * c.yDistCoefficient

    def getSkeletonImg(self):
        return self.skeletonImg

    def getSkeletonId(self):
        return self.id

    def getSkeletonKeypoints(self):
        return self.lastKeypoints


def normalize(keypoints, boundingBox):
    bbDims = [boundingBox[0][0] - boundingBox[0][1],
              boundingBox[1][0] - boundingBox[1][1],
              boundingBox[2][0] - boundingBox[2][1]]

    try:
        return [[(i[0] - boundingBox[0][1]) / bbDims[0],
                 (i[1] - boundingBox[1][1]) / bbDims[1],
                 (i[2] - boundingBox[2][1]) / bbDims[2]]
                for i in keypoints]
    except ZeroDivisionError:
        return [[(i[0] - boundingBox[0][1]) / (bbDims[0] + 1),
                 (i[1] - boundingBox[1][1]) / (bbDims[1] + 1),
                 (i[2] - boundingBox[2][1]) / (bbDims[2] + 1)]
                for i in keypoints]


# function estimates not detected keypoints in rgbd space by symmetry
def estimateNotDetectedKeypoints(human):
    mirrorNotDetectedKeypoints(human)
    parentizeNotDetectedKeypoints(human)
    return human


def mirrorNotDetectedKeypoints(human):
    headA = 0.5  # distance between head and neck will be a times distance between neck and hips
    # head
    if not detectedOrInterpolated(human[0]):
        human[0] = pointSym(human, x=8, o=1, a=headA)
    # shoulders and hips
    # if one shoulder (or hip) is detected and second is not
    # then second will be inverted through a middle point (neck/middle hip)
    for i in (2, 5, 1), (5, 2, 1), (9, 12, 8), (12, 9, 8):
        if not detectedOrInterpolated(human[i[0]]) and detectedOrInterpolated(human[i[1]]):
            human[i[0]] = pointSym(human, x=i[1], o=i[2], a=1)
    # arms and legs
    # tuple is like: ( not detected kp, detected kp, parent of not detected, parent of detected )
    # parents ar shoulders for arms and hips for legs
    for i in (3, 6, 2, 5), (4, 7, 2, 5), (6, 3, 5, 2), (7, 4, 5, 2), \
             (13, 10, 12, 9), (14, 11, 12, 9), (10, 13, 9, 12), (11, 14, 9, 12):
        if not detectedOrInterpolated(human[i[0]]) and detectedOrInterpolated(human[i[1]]):
            diff = [human[i[3]][j] - human[i[2]][j] for j in range(3)]
            human[i[0]] = [human[i[1]][j] - diff[j] for j in range(3)] + [1.0]


def parentizeNotDetectedKeypoints(human):
    for kp in 2, 5, 9, 12, 3, 6, 4, 7, 10, 13, 11, 14:
        if not detectedOrInterpolated(human[kp]):
            if kp == 5:
                parent = 1
            elif kp == 12:
                parent = 8
            else:
                parent = kp - 1
            human[kp] = human[parent]


# x - index of point to reflect
# o - index of symmetry point
# a - distance between counted point and o be 'a' times distance between x and o
def pointSym(human, x, o, a):
    return [int((human[o][0] - human[x][0]) * a) + human[o][0],
            int((human[o][1] - human[x][1]) * a) + human[o][1],
            int((human[o][2] - human[x][2]) * a) + human[o][2],
            1.0]


# function returns whether keypoint is detected or interpolated (but not correctly)
def detectedOrInterpolated(kp):
    return kp[-1] >= c.keypointThreshold or kp[-1] == -1
