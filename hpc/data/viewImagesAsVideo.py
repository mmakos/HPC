import argparse
import os
import sys
from time import time

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("images_folder", help="Folder to your data images relative to /data/images.")
parser.add_argument("-s", "--skeleton", type=int, default=0, help="Which skeleton you want to view.")
parser.add_argument("-f", "--fps", type=int, help="Frames per second - default is as fast as possible.")
parser.add_argument("-z", "--zoom", type=int, default=8, help="Factor you want to multiply each dimension.")
args = parser.parse_known_args()[0]

if not os.path.isdir(args.images_folder):
    args.images_folder = "../../data/images/" + args.images_folder
    if not os.path.isdir(args.images_folder):
        print("No such folder. Please make sure you typed correct directory path.")
        exit()

t = time()
for i in range(sys.maxsize):
    img = cv2.imread(f"{args.images_folder}/f{i}s{args.skeleton}.png")
    if img is None:
        img = np.zeros((16, 32))
    img = cv2.resize(img, (args.zoom * img.shape[1], args.zoom * img.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imshow(f"Skeleton {args.skeleton}", img)
    if args.fps:
        while time() - t < 1 / args.fps:
            pass
        t = time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
