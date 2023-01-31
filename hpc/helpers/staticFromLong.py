import os

frames = 1

for pose in ("lie", "stand", "sit", "lean", "kneel"):
    try:
        os.mkdir(f"data/images/rs/48frames/{pose}/static")
    except FileExistsError:
        pass

    for v in range(20):
        if not os.path.isfile(f"data/images/rs/48frames/{pose}/{pose}{v}.png"):
            continue
        os.system(f"python hpc/data/getStaticImages.py rs/48frames/{pose}/{pose}{v}.png -f {frames}")
        for f in range(frames):
            try:
                os.remove(f"data/images/rs/48frames/{pose}/static/{pose}{v}_{f}.png")
            except FileNotFoundError:
                pass
            os.rename(f"data/images/rs/48frames/{pose}/{pose}{v}_{f}.png", f"data/images/rs/48frames/{pose}/static/{pose}{v}_{f}.png")
