# general
frameWidth = 640
frameHeight = 480
depthWidth = 640
depthHeight = 480
framesNumber = 32
keypointsNumber = 15

# keypoints detection
keypointThreshold = 0.4  # below this threshold, keypoints from openPose are not considered
minDetectedKeypoints = 8  # minimum amount of detected keypoints for classifying pose
interpolateNotDetected = True  # Interpolates not detected keypoints with previous and following frame
fillNotDetected = True  # fills not detected keypoints with several algorithms (gives better accuracy)
# Parents of keypoints (parent of keypoint 'i' is 'keypointsParents[i]')
keypointsParents = (1, 2, 4, 4, -1, 4, 5, 6, 9, 10, 11, -1, 11, 12, 13)

# normalization
# If true, skeletons normalization will be normalized to square (+ depth independent), not rectangle
squareNormalization = False

# tracking
maxDeltaCoefficient = 0.5  # F
probThreshold = 0.1  # probability to consider two skeletons as the same skeleton
frameTime = 0.333  # 10 FPS
maxFrameTime = 1 / 5  # if live frame time is greater, then pose will estimated for this value, (interpolation for 1FPS makes no sense)
inputFrameRate = 30  # frame rate of input (for proceeding and estimating)
outputFrameRate = 10  # frame rate of output dataset (for proceeding)

# dataset creating
minLongImageLength = framesNumber

# dataset augmentation
# make small
imgFrameRate = 15  # frame rate of original train images
minOutputFrameRate = 15  # train images min frame rate
maxOutputFrameRate = 15  # train images max frame rate
frameRateStep = 1  # e.g. if 1 images will have frame rates: 5, 6, 7, 8, ..., 15
nextStartStep = 8  # e.g. for 4 image will be taken from frame 0-32, 4-36, 8-40 itd.
maxUpDownRotationAngle = 30

# training
batchSize = 64
epochs = 20
learningRate = 0.0001

# hybrid
distancePoints = (8, 9, 10, 12, 13, 14)  # tuple of points, which defines distance to categorize pose to dyn or static
statDynThreshold = 2700  # distance below which poses are static and above - dynamic
xDistCoefficient = 1.5
yDistCoefficient = 0.5

# poses
poses = (
    "stand",
    "sit",
    "lie",
    "lean",
    "kneel",
    "walk",
    "jump",
)

# used for estimation depth of not detected keypoints on depth canal
connections = (
    [1],
    (0, 2),
    (1, 4),
    [4],
    (2, 3, 5, 11),
    (4, 6),
    (5, 7),
    [6],
    [9],
    (8, 10),
    (9, 11),
    (10, 12, 4),
    (11, 13),
    (12, 14),
    [13]
)

keypoints = (
    "Right wrist",
    "Right elbow",
    "Right shoulder",
    "Nose",
    "Neck",
    "Left shoulder",
    "Left elbow",
    "Left wrist",
    "Left ankle",
    "Left knee",
    "Left hip",
    "Middle hip",
    "Right hip",
    "Right knee",
    "Right ankle",
)

# Keypoints order mapping from openpose format to my
openPoseKeypointsOrder = (4, 3, 2, 0, 1, 5, 6, 7, 14, 13, 12, 8, 9, 10, 11)

rightWrist = 0
rightElbow = 1
rightShoulder = 2
nose = 3
neck = 4
leftShoulder = 5
leftElbow = 6
leftWrist = 7
leftAnkle = 8
leftKnee = 9
leftHip = 10
middleHip = 11
rightHip = 12
rightKnee = 13
rightAnkle = 14
