import cv2
import sys
import os

import poses

dir_path = os.path.dirname( os.path.realpath( __file__ ) )
sys.path.append( dir_path + '/../../externals/openpose/build/python/openpose/Release' )
os.environ[ 'PATH' ] = os.environ[ 'PATH' ] + ';' + dir_path + '/../../externals/openpose/build/x64/Release;' + dir_path + '/../../externals/openpose/build/bin;'
import pyopenpose as op

image = cv2.imread( '../../img/ladies.jpg' )
cv2.imshow( "Human Pose Classification", image )
cv2.waitKey( 0 )


def git():
    return [ i for i in poses.poses ]


print( git() )
print( f"{ 0 } ania { 2 }")