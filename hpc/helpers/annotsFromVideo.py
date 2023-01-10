import os

pose = "lie"

for v in range(12, 100):
    if os.path.isdir(f"../../data/videos/rs/{pose}{v}"):
        os.system(f"python ../data/proceedVideo.py rs/{pose}{v}/ -p rs/{pose} -k")
        os.rename(f"../../data/images/rs/{pose}/s0at0.p", f"../../data/images/rs/{pose}/{pose}{v}at0.p")
