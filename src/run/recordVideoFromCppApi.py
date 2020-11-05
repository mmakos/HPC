import cv2
import sys
import consts as c
from time import time

videoRGB = cv2.VideoWriter( sys.argv[ 1 ] + "RGB.mp4", -1, 10.0, ( c.frameWidth, c.frameHeight ) )
# videoDepth = cv2.VideoWriter( sys.argv[ 1 ] + "Depth.mp4", -1, 10.0, ( c.frameWidth, c.frameHeight ), False )
# frameRGB = cv2.cvtColor( np.array( stream.getTestRGB(), dtype = np.uint8 ), cv2.COLOR_RGB2BGR )
frameTime = time()

while True:
    # frameDepth = np.array( stream.getTestDepth(), dtype = np.uint8 )
    # videoRGB.write( frameRGB )
    # videoDepth.write( frameDepth )

    # fps = 1.0 / ( time() - frameTime )
    # frameTime = time()
    # cv2.putText( frameRGB, "FPS: " + "{:.3f}".format( fps ), ( 5, 23 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ( 255, 0, 0 ), 2 )
    # cv2.imshow( 'frame RGB', frameRGB )
    if cv2.waitKey( 1 ) & 0xFF == ord( 'q' ):
        break

#videoDepth.release()
videoRGB.release()
cv2.destroyAllWindows()
