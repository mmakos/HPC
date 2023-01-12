import os

import hpc.consts as c

for pose in "kneel", "lean", "lie", "sit", "walk":
    os.system(f"python ../data/augment.py rs/{pose}/static -r 800 -o rsRot/{pose}_{c.poses.index(pose)}")
