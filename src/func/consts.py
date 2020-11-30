frameWidth = 640
frameHeight = 480
depthWidth = 640
depthHeight = 480
frameDepth = 10000    # to set

framesNumber = 32
keypointsNumber = 15

# keypoints detection
keypointThreshold = 0.1
minDetectedKeypoints = 5

# tracking
maxDeltaCoefficient = 0.2     # F
probThreshold = 0.1
frameTime = 0.333       # 10 FPS
maxFrameTime = 1 / 5    # if live frame time is greater, then pose will estimated for this value, (interpolation for 1FPS makes no sense)
# inputFrameRate = 30     # frame rate of input (for proceeding and estimating)
outputFrameRate = 10    # frame rate of output dataset (for proceeding)

# dataset augmentation
imgFrameRate = 30           # frame rate of original train images
minOutputFrameRate = 5      # train images min frame rate
maxOutputFrameRate = 15     # train images max frame rate
frameRateStep = 1           # e.g. if 1 images will have frame rates: 5, 6, 7, 8, ..., 15

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

# training
batchSize = 32

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
