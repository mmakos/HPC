import pickle
import os

path = "../../data/images/rs/walk/"


def readSkels():
    skeletons = []
    beginFrames = []
    keysFiles = os.listdir( path )
    for keysFile in keysFiles:
        if keysFile[ -2: ] == ".p":
            skeletons.append( pickle.load( open( path + keysFile, "rb" ) ) )
            beginFrames.append( int( keysFile.split( 'at' )[ 1 ].split( ".p" )[ 0 ] ) )
    return skeletons, beginFrames


print( readSkels() )