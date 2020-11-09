import cv2


def displayFrameTime( img, sec ):
    cv2.putText( img, "Time: " + "{:.3f}s".format( sec ), ( 5, 23 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ( 255, 0, 0 ), 2 )