frameWidth = 640
frameHeight = 480
depthWidth = 640
depthHeight = 480
frameDepth = 100    # to set

framesNumber = 32
keypointsNumber = 15

# keypoints detection
keypointThreshold = 0.1

# tracking
minDeltaCoefficient = 0.01
probThreshold = 0.01

# poses
poses = (
    "stand",
    "walk",
    "run",
    "sit",
    "lie",
    "dance"
)

# training
batchSize = 8
