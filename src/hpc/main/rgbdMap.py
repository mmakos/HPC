import src.consts as c


# function maps given keypoints to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD( keypoints, depthCanal ):
    keypointsRGBD = []
    for i in range( c.keypointsNumber ):
        if keypoints[ i ][ 2 ] >= c.keypointThreshold:        # keypoint is detected
            keypointsRGBD.append( [ int( keypoints[ i ][ 0 ] ), int( keypoints[ i ][ 2 ] ),
                                  depthCanal[ int( keypoints[ i ][ 0 ] ), int( keypoints[ i ][ 1 ] ) ] ] )
        else:
            keypointsRGBD.append( [ 0.0, 0.0, 0.0 ] )
