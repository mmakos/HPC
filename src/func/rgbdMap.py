import consts as c


# function maps given keypoints of all humans to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD( keypoints, depthCanal ):
    keypointsRGBD = []
    for human in keypoints:
        humanRGBD = []
        for i in range( c.keypointsNumber ):
            if human[ i ][ 2 ] >= c.keypointThreshold:        # keypoint is detected
                humanRGBD.append( [ int( human[ i ][ 0 ] + 0.5 ), int( human[ i ][ 1 ] + 0.5 ),
                                    depthCanal[ int( human[ i ][ 1 ] * c.depthHeight / c.frameHeight + 0.5 ),
                                                int( human[ i ][ 0 ] * c.depthWidth / c.frameWidth + 0.5 ) ] ] )
            else:
                humanRGBD.append( [ 0.0, 0.0, 0.0 ] )
        keypointsRGBD.append( humanRGBD )
    return keypointsRGBD
