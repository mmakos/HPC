import numpy as np

dataset = "All.npz"
new_pose_order = (0, 1, 2, 3, 4, 5, -1, 6)

with np.load("data/datasets/train/" + dataset, allow_pickle=True) as data:
    img = data['images']
    lab = data['labels']
    out_lab = np.zeros(lab.shape)

    for i, l in enumerate(lab):
        out_lab[i] = new_pose_order[l]

    np.savez_compressed('data/datasets/train/' + dataset[:-4] + "exp.npz", images=img, labels=out_lab)

