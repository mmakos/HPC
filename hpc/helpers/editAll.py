import os

pose = "lie"

for a in range(8, 100):
    if os.path.isfile(f"data/images/rs/{pose}/{pose}{a}at0.p"):
        os.system(f"python hpc/data/fillKeypoints.py rs/{pose}{a}/ -p rs/{pose}/{pose}{a}at0.p")
        os.remove(f"data/images/rs/{pose}/{pose}{a}at0.p")
        os.rename(f"data/images/rs/{pose}/{pose}{a}at0_f.p", f"data/images/rs/{pose}/{pose}{a}at0.p")
