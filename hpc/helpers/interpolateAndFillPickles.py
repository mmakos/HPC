import os
import pickle

import numpy as np

from hpc.consts import keypointsNumber, keypointThreshold, poses
from hpc.core.skeleton import estimateNotDetectedKeypoints


def interpolateNotDetectedKeypoints(skeletonImg, images_count):
    for kp in range(keypointsNumber):
        begin = -1
        ranges = list()
        for i in range(images_count):
            if skeletonImg[i, kp, 3] < keypointThreshold:
                if begin == -1:
                    begin = i
            elif begin >= 0:
                ranges.append((begin, i))
                begin = -1
        if begin > 0:
            ranges.append((begin, images_count))

        for r in ranges:
            for i in range(r[0], r[1]):
                if r[0] == 0:
                    skeletonImg[i, kp] = skeletonImg[r[1], kp]
                    skeletonImg[i, kp, 3] = -1
                elif r[1] == images_count:
                    skeletonImg[i, kp] = skeletonImg[r[0] - 1, kp]
                    skeletonImg[i, kp, 3] = -1
                else:
                    skeletonImg[i, kp] = skeletonImg[r[0] - 1, kp] + (
                            skeletonImg[r[1], kp] - skeletonImg[r[0] - 1, kp]) * (i - r[0] + 1) / (
                                                      r[1] - r[0] + 1)
                    skeletonImg[i, kp, 3] = 1


for skel in range(20):
    for pose in poses:
        if os.path.isfile(f"data/images/rs/originManFilled/{pose}/{pose}{skel}at0.p"):
            skeletons = pickle.load(open(f"data/images/rs/originManFilled/{pose}/{pose}{skel}at0.p", "rb"))

            skels = np.asarray(skeletons)
            interpolateNotDetectedKeypoints(skels, skels.shape[0])

            pickle.dump(skels.tolist(), open(f"data/images/rs/originManFilled/{pose}/{pose}{skel}at0.p", "wb"))
