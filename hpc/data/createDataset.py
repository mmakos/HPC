import argparse
import os

import cv2
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

import hpc.consts as c
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument("poses", help="Path to folder with your poses folders relative to /data/images.")
parser.add_argument("-d", "--dataset_name", help="Path to output dataset file relative to /data/datasets.")
parser.add_argument("-a", "--amount", type=int, help="amount of samples in each label (same).")
args = parser.parse_known_args()[0]

path = "data/images/" + args.poses
if args.dataset_name is None:
    args.dataset_name = args.poses
dsPath = "data/datasets/" + args.dataset_name

datasetImages = []
datasetLabels = []
label = 0
for _, poses, _ in os.walk(path):
    for pose in poses:
        posePath = path + "/" + pose
        try:
            label = int(pose.split("_")[-1])
            x = c.poses[label]
        except (IndexError, ValueError):
            continue
        for _, _, images in os.walk(posePath):
            if args.amount is not None:
                while len(images) > args.amount:
                    images.pop(randint(0, len(images) - 1))
            for i in tqdm(range(len(images)), desc=c.poses[label]):
                imgPath = posePath + "/" + images[i]
                img = cv2.imread(imgPath)
                datasetImages.append(img)
                datasetLabels.append(label)

datasetImages = np.array(datasetImages)
datasetLabels = np.array(datasetLabels)
datasetImages, datasetLabels = shuffle(datasetImages, datasetLabels)
np.savez_compressed(dsPath, images=datasetImages, labels=datasetLabels)
print("\nDataset created and saved to file data/datasets/" + args.dataset_name + ".npz")
