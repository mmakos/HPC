import src.consts as c


# function maps given keypoints to RGBD image
# keypoints are [ x cord, y cord, score ]
def mapToRGBD( keypoints, depthCanal ):
    keypointsRGBD = []
    for keypoint in keypoints:
        if keypoint[ 2 ] >= c.keypointThreshold:        # keypoint is detected
            keypointsRGBD.append( depthCanal[ int( keypoint[ 0 ] ), int( keypoint[ 1 ] ) ] )
        else:
            keypointsRGBD.append( 0.0 )
