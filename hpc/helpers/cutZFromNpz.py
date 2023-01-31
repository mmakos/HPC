import numpy as np
from tqdm import tqdm

name = "train/All.npz"
outName = "train/AllNoDepth.npz"

with np.load("data/datasets/" + name, allow_pickle=True) as data:
    images = data['images']
    labels = data['labels']

for i in tqdm(range(len(images)), desc="Images"):
    images[i, :, :, 0] = np.zeros([images[i].shape[0], images[i].shape[1]])

np.savez_compressed("data/datasets/" + outName, images=images, labels=labels)
