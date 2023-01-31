import os

from hpc.consts import poses

directory = "16frames"

for pose in poses:
    for i in range(20):
        if os.path.isfile(f"data/images/rs/{directory}/{pose}/{pose}{i}at0.p") and not os.path.isfile(f"data/images/rs/{directory}/{pose}/{pose}{i}.png"):
            os.system(f"python hpc/data/proceedVideo.py ../../dataOrigin/videos/rs/{pose}{i}/ -a rs/{directory}/{pose}/{pose}{i}at0.p -p rs/{directory}/{pose}/{pose} -l")
            os.rename(f"data/images/rs/{directory}/{pose}/{pose}/s0.png", f"data/images/rs/{directory}/{pose}/{pose}{i}.png")

# for starFrame in (0, 355, 896, 1788):
#     for directory in ["apInterpolationSquaredNormalization"]:
#         for pose in poses:
#             if os.path.isfile(f"data/images/orbbec_20NOV/{directory}/{pose}at{starFrame}.p"):
#                 os.system(f"python hpc/data/proceedVideo.py ../../dataOrigin/videos/orbbec_20NOV/{pose}/ -a orbbec_20NOV/{directory}/{pose}at{starFrame}.p -p orbbec_20NOV/{directory}/{pose} -l")
#                 os.rename(f"data/images/orbbec_20NOV/{directory}/{pose}/s0.png", f"data/images/orbbec_20NOV/{directory}/{pose}at{starFrame}.png")
