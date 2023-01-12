import os

import cv2
import numpy as np

from hpc.consts import poses

for pose in poses:
    inputFile = f"data/images/orbbec_20NOV/apOriginal/{pose}.png"
    outputDir = inputFile[:-4]

    os.mkdir(outputDir)
    frames = cv2.imread(inputFile)

    for i in range(frames.shape[1]):
        if i + 64 < frames.shape[1]:
            img = [frames[:, i + j] for j in range(0, 64, 2)]
            cv2.imwrite(f"{outputDir}/f{i}s0.png", np.swapaxes(np.asarray(img), 0, 1))
