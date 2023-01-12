import os

from hpc.consts import poses

for pose in poses:
    if os.path.isfile(f"data/images/orbbec_20NOV/apFilled/{pose}at0_f.p"):
        os.system(f"python hpc/data/proceedVideo.py orbbec_20NOV/{pose}/ -a orbbec_20NOV/apFilled/{pose}at0_f.p -p orbbec_20NOV/apFilled/{pose}")
        # os.rename(f"data/images/orbbec_20NOV/apFilled/{pose}/s0.png", f"data/images/orbbec_20NOV/apFilled/{pose}/{pose}.png")
