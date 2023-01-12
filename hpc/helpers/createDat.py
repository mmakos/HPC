import argparse
import os
import shutil
from random import shuffle

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", help="Path to output dataset file relative to /data/datasets.")
parser.add_argument("-p", "--poses", default="poses", help="Path to folder with your poses folders relative to /data.")
parser.add_argument("-z", "--zipped", help="File will be compressed to .zip file.", action="store_true")
args = parser.parse_known_args()[0]

path = "data/" + args.poses

dataset = []
label = 0
for _, poses, _ in os.walk(path):
    for pose in poses:
        posePath = path + "/" + pose
        label = pose.split("_")[1]
        for _, _, images in os.walk(posePath):
            for image in images:
                imgPath = posePath + "/" + image
                print(imgPath + "\tlabel = " + label + "\t- done.")
                img = cv2.imread(imgPath)
                dataset.append([np.array(img), label])

shuffle(dataset)
dsPath = "data/datasets/" + args.dataset_name
np.save(dsPath, dataset)
if not args.zipped:
    print("\nDataset created and saved to file /data/datasets/" + args.dataset_name + ".npy")
else:
    os.mkdir(dsPath)
    os.rename(dsPath + ".npy", dsPath + "/" + args.dataset_name + ".npy")
    shutil.make_archive(dsPath, 'zip', dsPath)
    os.remove(dsPath + "/" + args.dataset_name + ".npy")
    os.rmdir(dsPath)
    print("\nDataset compressed and save to file /data/datasets/" + args.dataset_name + ".zip")
