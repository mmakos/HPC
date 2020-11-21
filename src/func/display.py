import cv2
import consts as c
import frame


def displayFrameTime( img, sec ):
    cv2.putText( img, "Time: " + "{:.3f}s".format( sec ), ( 5, 23 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ( 255, 0, 0 ), 2 )


def displayPose( img, keypoints, pose ):
    keypoints = [ i for i in keypoints if i[ 3 ] >= c.keypointThreshold ]
    bb = frame.getBoundingBox( keypoints )
    x = bb[ 0 ][ 1 ] - 10
    y = bb[ 1 ][ 1 ] - 10
    x2 = bb[ 0 ][ 0 ] + 10
    y2 = bb[ 1 ][ 0 ] + 10
    x = 0 if x < 0 else x
    y = 0 if y < 0 else y
    cv2.rectangle( img, ( x, y ), ( x2, y2 ), ( 0, 255, 0 ), 2 )
    ( textW, textH ) = cv2.getTextSize( pose, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2 )[ 0 ]
    cv2.rectangle( img, ( x, y ), ( x + textW + 2, y + textH + 2 ), ( 0, 255, 0 ), cv2.FILLED )
    cv2.putText( img, pose, ( x, y + textH - 1 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 0, 0, 255 ), 2 )
