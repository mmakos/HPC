# general
frameWidth = 640
frameHeight = 480
depthWidth = 640
depthHeight = 480
framesNumber = 32
keypointsNumber = 15

# keypoints detection
keypointThreshold = 0.1     # below this threshold, keypoints from openPose are not considered
minDetectedKeypoints = 8    # minimum amount of detected keypoints for classifying pose
fillNotDetected = True      # fills not detected keypoints with several algorithms (gives better accuracy)

# tracking
maxDeltaCoefficient = 0.5     # F
probThreshold = 0.1     # probability to consider two skeletons as the same skeleton
frameTime = 0.333       # 10 FPS
maxFrameTime = 1 / 5    # if live frame time is greater, then pose will estimated for this value, (interpolation for 1FPS makes no sense)
inputFrameRate = 30     # frame rate of input (for proceeding and estimating)
outputFrameRate = 10    # frame rate of output dataset (for proceeding)

# dataset creating
minLongImageLength = framesNumber

# dataset augmentation
# make small
imgFrameRate = 15           # frame rate of original train images
minOutputFrameRate = 15     # train images min frame rate
maxOutputFrameRate = 15     # train images max frame rate
frameRateStep = 1           # e.g. if 1 images will have frame rates: 5, 6, 7, 8, ..., 15
nextStartStep = 8           # e.g. for 4 image will be taken from frame 0-32, 4-36, 8-40 itd.
maxUpDownRotationAngle = 30

# training
batchSize = 64
epochs = 50
learningRate = 0.0001

# hybrid
distancePoints = ( 9, 10, 11, 12, 13, 14 )  # tuple of points, which defines distance to categorize pose to dyn or static
statDynThreshold = 1695     # distance below which poses are static and above - dynamic
xDistCoefficient = 1
yDistCoefficient = 1

# poses
poses = (
    "stand",
    "sit",
    "lie",
    "lean",
    "kneel",
    "walk",
    "run",
    "jump",
    "dance",
)

# used for estimation depth of not detected keypoints on depth canal
connections = (
    [ 1 ],
    ( 0, 2, 5, 8 ),
    ( 1, 3 ),
    ( 2, 4 ),
    [ 3 ],
    ( 1, 6 ),
    ( 5, 7 ),
    [ 6 ],
    ( 1, 9, 12 ),
    ( 8, 10 ),
    ( 9, 11 ),
    [ 10 ],
    ( 8, 13 ),
    ( 12, 14 ),
    [ 13 ]
)

keypoints = (
    "Nose",
    "Neck",
    "Right shoulder",
    "Right elbow",
    "Right wrist",
    "Left shoulder",
    "Left elbow",
    "Left wrist",
    "Middle hip",
    "Right hip",
    "Right knee",
    "Right ankle",
    "Left hip",
    "Left knee",
    "Left ankle",
)
