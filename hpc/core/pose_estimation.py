import os
import sys

import hpc.consts as c


class PoseEstimation:
    def __init__(self, estimationLibrary="AlphaPose", addParams=None):
        self.__initEstimationLibrary(estimationLibrary, addParams)

    def estimatePose(self, frameRGB):
        if self.estimationLibrary == "OpenPose":
            import pyopenpose as op
            datum = op.Datum()
            datum.cvInputData = frameRGB
            self.opWrapper.emplaceAndPop([datum])
            return frameRGB, datum.poseKeypoints
        else:
            pose = self.apWrapper.process("unnamed", frameRGB)
            return self.apWrapper.vis(frameRGB, pose), self.__alphaPoseKeypoints(pose)


    def __alphaPoseKeypoints(self, pose):
        skeletons = pose["result"]
        humansKeypoints = []

        for skeleton in skeletons:
            keypoints = skeleton["keypoints"].tolist()
            kpScores = skeleton["kp_score"].tolist()
            for i, _ in enumerate(keypoints):
                keypoints[i].append(kpScores[i][0])
            humansKeypoints.append(self.__alphaPoseKeypointsOrder(keypoints))
        return humansKeypoints

    def __alphaPoseKeypointsOrder(self, keypoints: list):
        keypointsOredered = [keypoints[i] for i in (0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3)]
        keypointsOredered.insert(1, self.__interpolatePoints(keypoints[5], keypoints[6]))
        keypointsOredered.insert(8, self.__interpolatePoints(keypoints[11], keypoints[12]))
        return keypointsOredered

    def __interpolatePoints(self, firstPoint, secondPoint):
        return [(f + s) / 2 for f, s in zip(firstPoint, secondPoint)]

    def __initEstimationLibrary(self, estimationLibrary, addParams):
        self.estimationLibrary = estimationLibrary
        if estimationLibrary == "OpenPose":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            sys.path.append(dir_path + '/../../externals/openpose/build/python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../externals/openpose/build/x64/Release;' + dir_path + '/../../externals/openpose/build/bin;'
            self.__initOpenPose(self.__getOpParams(addParams))
        else:
            from hpc.run.alpha_pose_api import SingleImageAlphaPose
            addParams.append("--format")
            addParams.append("open")
            self.apWrapper = SingleImageAlphaPose(addParams)

    def __initOpenPose(self, opParams):
        # starting OpenPose
        if not opParams:
            opParams = dict()
        opParams["model_folder"] = "../../externals/openpose/models/"
        opParams["render_threshold"] = c.keypointThreshold
        import pyopenpose as op
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(opParams)
        self.opWrapper.start()

    def __getOpParams(self, allArgs):
        params = dict()
        for param in range(0, len(allArgs)):
            curr_item = allArgs[param]
            if param != len(allArgs) - 1:
                next_item = allArgs[param + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params:
                    params[key] = next_item
        return params
