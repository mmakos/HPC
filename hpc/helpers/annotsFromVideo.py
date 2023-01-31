import os

from hpc.consts import poses

for pose in poses:
    if os.path.isdir(f"data/videos/orbbec_20NOV"):
        os.system(f"python hpc/data/proceedVideo.py orbbec_20NOV/{pose}/ -p orbbec_20NOV/apAllOriginalNoFill/{pose} -k  --checkpoint external/AlphaPose/pretrained_models/fast_res50_256x192.pth --cfg external/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
        os.rename(f"data/images/orbbec_20NOV/apAllOriginalNoFill/{pose}/s0at0.p", f"data/images/orbbec_20NOV/apAllOriginalNoFill/{pose}at0.p")
