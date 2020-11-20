frameWidth = 640
frameHeight = 480
depthWidth = 640
depthHeight = 480
frameDepth = 10000    # to set

framesNumber = 32
keypointsNumber = 15

# keypoints detection
keypointThreshold = 0.1
minDetectedKeypoints = 12

# tracking
maxDeltaCoefficient = 0.1     # F
probThreshold = 0.1
frameTime = 0.333    # 10 FPS
maxFrameTime = 0.1  # 10 FPS

# poses
poses = (
    "stand",
    "walk",
    "run",
    "sit",
    "lie",
    "dance",
    "jump"
)

# training
batchSize = 8
