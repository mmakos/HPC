import os

import hpc.consts as c

# for pose in ("lie", "stand", "sit", "lean", "kneel"):
for pose in ("jump", "walk"):
    os.system(f"python hpc/data/augment.py rs/16frames/{pose}/{pose} -r 100 -o rs/16frames/{pose}_{c.poses.index(pose)}")
