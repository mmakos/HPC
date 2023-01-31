import os

import cv2
import numpy as np

from hpc.consts import poses

# for j in range(20):
#         for directory in ["48frames"]:
#             for pose in ["walk", "jump"]:
#                 if os.path.isfile(f"data/images/rs/{directory}/{pose}/{pose}{j}.png"):
#                     inputFile = f"data/images/rs/{directory}/{pose}/{pose}{j}.png"
#
#                     if not os.path.isdir(f"data/images/rs/{directory}/{pose}/{pose}"):
#                         os.mkdir(f"data/images/rs/{directory}/{pose}/{pose}")
#                     frames = cv2.imread(inputFile)
#
#                     for i in range(0, frames.shape[1], 4):
#                         if i + 48 < frames.shape[1]:
#                             img = [frames[:, i + j] for j in range(0, 48, 1)]
#                             cv2.imwrite(f"data/images/rs/{directory}/{pose}/{pose}/f{i}s{j}.png", np.swapaxes(np.asarray(img), 0, 1))

for starFrame in (0, 355, 896, 1788):
    for directory in ["apInterpolationFrame48"]:
        for pose in poses:
            if os.path.isfile(f"data/images/orbbec_20NOV/{directory}/{pose}at{starFrame}.png"):
                inputFile = f"data/images/orbbec_20NOV/{directory}/{pose}at{starFrame}.png"

                if not os.path.isdir(f"data/images/orbbec_20NOV/{directory}/{pose}_{poses.index(pose)}"):
                    os.mkdir(f"data/images/orbbec_20NOV/{directory}/{pose}_{poses.index(pose)}")
                frames = cv2.imread(inputFile)

                for i in range(frames.shape[1]):
                    if i + 96 < frames.shape[1]:
                        img = [frames[:, i + j] for j in range(0, 96, 2)]
                        cv2.imwrite(f"data/images/orbbec_20NOV/{directory}/{pose}_{poses.index(pose)}/f{i + starFrame}s0.png", np.swapaxes(np.asarray(img), 0, 1))
