import tensorflow as tf

from hpc.core.pose_estimation import PoseEstimation
from hpc.core.frame import Frame
from hpc.core.preprocess import preprocess
import hpc.consts as c
import numpy as np
import hpc.core.display as display
from time import time


class Wrapper:
    # model is name of model
    def __init__(self, model, gpuMode=False, estimationLibrary="AlphaPose", addParams=None):
        self.model = None
        self.dynModel = None
        self.opWrapper = None
        self.frameNumber = 0
        self.time = time()
        self.__getModel(model, gpuMode)
        self.frame = Frame(self.model, self.dynModel)
        self.poseEstimation = PoseEstimation(estimationLibrary, addParams)

    def __getModel(self, model, gpuMode):
        if not gpuMode:
            tf.config.list_physical_devices('GPU')
            try:
                # Disable all GPUs
                tf.config.set_visible_devices([], 'GPU')
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != 'GPU'
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        if type(model) is str or (type(model) is tuple and len(model) == 1):  # one model
            modName = model
            dynName = None
        else:  # two models
            modName = model[0]
            dynName = model[1]

        self.model = tf.keras.models.load_model('data/models/' + modName)
        print("Model " + modName + " loaded.")
        print(self.model.summary())
        if dynName is not None:
            self.model = tf.keras.models.load_model('data/models/' + dynName)
            print("Model " + dynName + " loaded.")
            print(self.dynModel.summary())
        else:
            self.dynModel = None

    def proceed(self, frame, noDepth=False, noPose=False, noSkeleton=False, noTime=False):
        frameRGB, frameD = frame
        if self.frameNumber == 0:
            try:
                c.frameHeight, c.frameWidth, _ = frameRGB.shape
                c.depthHeight, c.depthWidth = frameD.shape
            except:
                pass

        frameRGBWithSkeletons, keypoints = self.poseEstimation.estimatePose(frameRGB)
        frameRGB = frameRGBWithSkeletons if not noSkeleton else frameRGB

        humans = preprocess(keypoints, frameD, noDepth)
        # convert frame to skeleton image
        poses = []
        if not noPose:
            poses = self.frame.proceedFrame(humans)
        for j, human in enumerate(humans):
            try:
                display.displayPose(frameRGB, human, str(poses[j][1]) + ": " +
                                    c.poses[np.argmax(poses[j][0])] +
                                    f" - {int(np.max(poses[j][0]) * 100)}%")
                display.skeleton(frameRGB, human)
            except:
                pass
        if not noTime:
            display.displayFrameTime(frameRGB, time() - self.time)
            self.time = time()

        self.frameNumber = self.frameNumber + 1
        return frameRGB, poses, humans
