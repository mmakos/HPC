import os

pose = "lie"
frames = 1

try:
    os.mkdir(f"../../data/images/rs/{pose}/static")
except FileExistsError:
    pass

for v in range(100):
    if not os.path.isfile(f"../../data/images/rs/{pose}/{pose}{v}.png"):
        continue
    os.system(f"python ../data/getStaticImages.py rs/{pose}/{pose}{v}.png -f {frames}")
    for f in range(frames):
        try:
            os.remove(f"../../data/images/rs/{pose}/static/{pose}{v}_{f}.png")
        except FileNotFoundError:
            pass
        os.rename(f"../../data/images/rs/{pose}/{pose}{v}_{f}.png", f"../../data/images/rs/{pose}/static/{pose}{v}_{f}.png")
