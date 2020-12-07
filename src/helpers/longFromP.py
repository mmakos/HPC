import os

pose = "sit"

for s in range( 100 ):
    if os.path.isfile( f"../../data/images/rs/{ pose }/{ pose }{ s }at0.p" ):
        os.system( f"python ../data/proceedVideo.py rs/{ pose }{ s }/ -a rs/{ pose }/{ pose }{ s }at0.p -p rs/{ pose } -l" )
        os.rename( f"../../data/images/rs/{ pose }/s0.png", f"../../data/images/rs/{ pose }/{ pose }{ s }.png" )