import consts as c
import statistics
from time import time


# function maps given keypoints of all humans to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD( keypoints, depthCanal ):
    keypointsRGBD = []
    try:
        for human in keypoints:
            humanRGBD = []
            pointsDetected = 0
            for i in range( c.keypointsNumber ):
                if human[ i ][ 2 ] >= c.keypointThreshold:        # keypoint is detected
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
                                        depthVal, human[ i ][ 2 ] ] )
                else:
                    humanRGBD.append( [ 0.0, 0.0, 0.0, 0.0 ] )
            # estimate not detected keypoints
            if pointsDetected >= c.minDetectedKeypoints:
                estimateDepthZeros( humanRGBD )
                keypointsRGBD.append( humanRGBD )
    except TypeError:
        pass
    return keypointsRGBD


# filter zeros from depth image
# point has to be already mapped to depth frame dimensions, ( x, y )!
# keyFunction is function, which you want to use to count point value (e.g. mean, max, min )
# don't set window value (it's only for recursion)
def filterDepthZeros( depthCanal, point, keyFunction=statistics.mean, window=1 ):
    win = int( window / 2 )     # win is how many pixels are in each direction from middle pixel
    pixels = []
    for x in range( point[ 0 ] - win, point[ 0 ] + win + 1 ):         # column of window
        for y in range( point[ 1 ] - win, point[ 1 ] + win + 1 ):     # row of window
            try:
                if x >= 0 and y >= 0:
                    px = depthCanal[ y ][ x ]        # frame dims are ( y, x )
                    if px > 0:
                        pixels.append( px )
            except IndexError:
                pass
    if len( pixels ) == 0:
        value = filterDepthZeros( depthCanal, point, keyFunction, window + 2 )
    else:
        value = keyFunction( pixels )
    return value


def estimateDepthZeros( points ):
    done = []
    for i, _ in enumerate( points ):
        if points[ i ][ 2 ] == 0:
            estimateDepthZeroPoint( i, points, done )


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
