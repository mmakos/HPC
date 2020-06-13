from time import time
from src.rgbdMap import mapToRGBD
from src.frame import Frame
import pyopenpose as op


# TODO - must return frame from RGBD video stream
# frame should be RGB image in cv2 format and separate depth canal
def getVideoFrame():
    return [[[]]], [[]]


# starting OpenPose
params = dict()
params[ "model_folder" ] = "../../models/"
opWrapper = op.WrapperPython()
opWrapper.configure( params )
opWrapper.start()

# starting frames proceed
frames = Frame()
frameTime = 0

while True:
    datum = op.Datum()
    frameRGB, frameD = getVideoFrame()
    datum.cvInputData = frameRGB
    opWrapper.emplaceAndPop( [ datum ] )

    # array of people with keypoints is in datum.poseKeypoints
    # proceedFrame gives for every human classified pose with score
    poses = frames.proceedFrame( mapToRGBD( datum.poseKeypoints ) )




    # to print FPS
    fps = 1.0 / ( time() - frameTime )
    frameTime = time()
