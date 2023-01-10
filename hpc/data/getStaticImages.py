import argparse

import cv2
import numpy as np

import consts as c

# Module loads images with encoded keypoints, takes one row, and stretch it to fill whole image (for static poses)

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Name of your image.")
parser.add_argument("-f", "--frames", type=int, default=1, help="How many frames to take from image.")
args = parser.parse_known_args()[0]

args.image = "../../data/images/" + args.image
image = cv2.imread(args.image)
for i in range(args.frames):
    outImg = np.zeros((image.shape[0], c.framesNumber, 3))
    for x in range(outImg.shape[0]):
        for y in range(outImg.shape[1]):
            outImg[x, y] = image[x, i, :]
    # frame = cv2.resize( frame, ( frame.shape[ 0 ], c.framesNumber ) )

    cv2.imwrite(f"{args.image[:-4]}_{i}.png", outImg)
