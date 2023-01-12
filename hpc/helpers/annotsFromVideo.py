import os

from hpc.consts import poses

for pose in poses:
    if os.path.isdir(f"data/videos/orbbec_20NOV"):
        os.system(f"python hpc/data/proceedVideo.py orbbec_20NOV/{pose}/ -p orbbec_20NOV/opAllOriginalNoFill/{pose} -k -e OpenPose")
        os.rename(f"data/images/orbbec_20NOV/opAllOriginalNoFill/{pose}/s0at0.p", f"data/images/orbbec_20NOV/opAllOriginalNoFill/{pose}at0.p")
