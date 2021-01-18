import consts as c
import statistics
import numpy as np


def preprocess( keypoints, depthCanal, noDepth=False ):
    try:
        getHead( keypoints )
        keypointsRGBD = mapToRGBD( keypoints, depthCanal, noDepth )
        return keypointsRGBD
    except TypeError:
        return []


# function maps given keypoints of all humans to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD( keypoints, depthCanal, noDepth=False ):
    keypointsRGBD = []
    for human in keypoints:
        humanRGBD = []
        pointsDetected = 0
        for i in range( c.keypointsNumber ):
            if det( human[ i ] ):        # keypoint is detected
                pointsDetected = pointsDetected + 1
                # if human[ i ][ 0 ] <= c.frameWidth and human[ i ][ 1 ] <= c.frameHeight:
                try:
                    point = ( int( human[ i ][ 0 ] * c.depthWidth / c.frameWidth + 0.5 ),
                              int( human[ i ][ 1 ] * c.depthHeight / c.frameHeight + 0.5 ) )
                    depthVal = depthCanal[ point[ 1 ] ][ point[ 0 ] ]
                    # depthVal = filterDepthZeros( depthCanal, point )
                except IndexError:  # when keypoint detected beyond the borders
                    depthVal = 0
                humanRGBD.append( [ int( human[ i ][ 0 ] + 0.5 ), int( human[ i ][ 1 ] + 0.5 ),
                                    int( depthVal ), human[ i ][ 2 ] ] )
            else:
                humanRGBD.append( [ 0.0, 0.0, 0.0, 0.0 ] )
        # estimate not detected keypoints
        if pointsDetected >= c.minDetectedKeypoints:
            if not noDepth:
                estimateDepthZeros( humanRGBD )
            keypointsRGBD.append( humanRGBD )
    return keypointsRGBD


# function estimate depth dimensions for points where depth value was not detected (is 0)
def estimateDepthZeros( points ):
    done = []
    for i, _ in enumerate( points ):
        if points[ i ][ 2 ] == 0:
            estimateDepthZeroPoint( i, points, done )


# function estimate depth dim for single point (called by above function)
def estimateDepthZeroPoint( i, points, done ):
    done.append( i )
    try:
        points[ i ][ 2 ] = statistics.mean( [ points[ j ][ 2 ] for j in c.connections[ i ] if points[ j ][ 2 ] > 0 ] )
    except statistics.StatisticsError:
        for j in c.connections[ i ]:
            if j not in done:
                estimateDepthZeroPoint( j, points, done )
        try:
            points[ i ][ 2 ] = statistics.mean( [ j for j in c.connections[ i ] if points[ j ][ 2 ] > 0 ] )
        except statistics.StatisticsError:
            points[ i ][ 2 ] = 1        # this happens only when all detected points has 0 depth


# function modifies nose as mean of head keypoints if nose is not detected
def getHead( keypoints ):
    for human in keypoints:
        if human[ 0 ][ -1 ] < c.keypointThreshold:   # nose not detected
            head = np.array( [ x for x in human[ 15:19 ] if not det( x ) ] )
            if len( head ) > 0:
                human[ 0 ] = head.mean( axis=0 ).tolist()
    return keypoints


# function estimates not detected keypoints in rgbd space by symmetry
def estimateNotDetectedKeypoints( human ):
    mirrorNotDetectedKeypoints( human )
    parentizeNotDetectedKeypoints( human )
    return human


def mirrorNotDetectedKeypoints( human ):
    headA = 0.5  # distance between head and neck will be a times distance between neck and hips
    # head
    if not det( human[ 0 ] ):
        human[ 0 ] = pointSym( human, x=8, o=1, a=headA )
    # shoulders and hips
    # if one shoulder (or hip) is detected and second is not
    # then second will be inverted through a middle point (neck/middle hip)
    for i in (2, 5, 1), (5, 2, 1), (9, 12, 8), (12, 9, 8):
        if not det( human[ i[ 0 ] ] ) and det( human[ i[ 1 ] ] ):
            human[ i[ 0 ] ] = pointSym( human, x=i[ 1 ], o=i[ 2 ], a=1 )
    # arms and legs
    # tuple is like: ( not detected kp, detected kp, parent of not detected, parent of detected )
    # parents ar shoulders for arms and hips for legs
    for i in (3, 6, 2, 5), (4, 7, 2, 5), (6, 3, 5, 2), (7, 4, 5, 2), \
             (13, 10, 12, 9), (14, 11, 12, 9), (10, 13, 9, 12), (11, 14, 9, 12):
        if not det( human[ i[ 0 ] ] ) and det( human[ i[ 1 ] ] ):
            diff = [ human[ i[ 3 ] ][ j ] - human[ i[ 2 ] ][ j ] for j in range( 3 ) ]
            human[ i[ 0 ] ] = [ human[ i[ 1 ] ][ j ] - diff[ j ] for j in range( 3 ) ] + [ 1.0 ]


def parentizeNotDetectedKeypoints( human ):
    for kp in 2, 5, 9, 12, 3, 6, 4, 7, 10, 13, 11, 14:
        if not det( human[ kp ] ):
            if kp == 5:
                parent = 1
            elif kp == 12:
                parent = 8
            else:
                parent = kp - 1
            human[ kp ] = human[ parent ]


# x - index of point to reflect
# o - index of symmetry point
# a - distance between counted point and o be 'a' times distance between x and o
def pointSym( human, x, o, a ):
    return [ int( ( human[ o ][ 0 ] - human[ x ][ 0 ] ) * a ) + human[ o ][ 0 ],
             int( ( human[ o ][ 1 ] - human[ x ][ 1 ] ) * a ) + human[ o ][ 1 ],
             int( ( human[ o ][ 2 ] - human[ x ][ 2 ] ) * a ) + human[ o ][ 2 ],
             1.0 ]


# function returns whether keypoint is detected (to make code shorter and more readable)
def det( kp ):
    return kp[ -1 ] >= c.keypointThreshold
