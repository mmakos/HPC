import statistics

import numpy as np

import hpc.consts as c


def preprocess(keypoints, depthCanal, noDepth=False):
    try:
        getHead(keypoints)
        keypointsRGBD = mapToRGBD(keypoints, depthCanal, noDepth)
        return keypointsRGBD
    except TypeError:
        return []


# function maps given keypoints of all humans to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD(keypoints, depthCanal, noDepth=False):
    keypointsRGBD = []
    for human in keypoints:
        humanRGBD = []
        pointsDetected = 0
        for i in range(c.keypointsNumber):
            if detected(human[i]):  # keypoint is detected
                pointsDetected = pointsDetected + 1
                # if human[ i ][ 0 ] <= c.frameWidth and human[ i ][ 1 ] <= c.frameHeight:
                try:
                    point = (int(human[i][0] * c.depthWidth / c.frameWidth + 0.5),
                             int(human[i][1] * c.depthHeight / c.frameHeight + 0.5))
                    depthVal = depthCanal[point[1]][point[0]]
                    # depthVal = filterDepthZeros( depthCanal, point )
                except IndexError:  # when keypoint detected beyond the borders
                    depthVal = 0
                humanRGBD.append([int(human[i][0] + 0.5), int(human[i][1] + 0.5),
                                  int(depthVal), human[i][2]])
            else:
                humanRGBD.append([0.0, 0.0, 0.0, 0.0])
        # estimate not detected keypoints
        if pointsDetected >= c.minDetectedKeypoints:
            if not noDepth:
                estimateDepthZeros(humanRGBD)
            keypointsRGBD.append(humanRGBD)
    return keypointsRGBD


# function estimate depth dimensions for points where depth value was not detected (is 0)
def estimateDepthZeros(points):
    done = []
    for i, _ in enumerate(points):
        if points[i][2] == 0:
            estimateDepthZeroPoint(i, points, done)


# function estimate depth dim for single point (called by above function)
def estimateDepthZeroPoint(i, points, done):
    done.append(i)
    try:
        points[i][2] = statistics.mean([points[j][2] for j in c.connections[i] if points[j][2] > 0])
    except statistics.StatisticsError:
        for j in c.connections[i]:
            if j not in done:
                estimateDepthZeroPoint(j, points, done)
        try:
            points[i][2] = statistics.mean([j for j in c.connections[i] if points[j][2] > 0])
        except statistics.StatisticsError:
            points[i][2] = 1  # this happens only when all detected points has 0 depth


# function modifies nose as mean of head keypoints if nose is not detected
def getHead(keypoints):
    for human in keypoints:
        if human[0][-1] < c.keypointThreshold:  # nose not detected
            head = np.array([x for x in human[15:19] if not detected(x)])
            if len(head) > 0:
                human[0] = head.mean(axis=0).tolist()
    return keypoints


# function returns whether keypoint is detected (to make code shorter and more readable)
def detected(kp):
    return kp[-1] >= c.keypointThreshold
