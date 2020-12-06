from tkinter import *
from PIL import Image, ImageTk
import pickle
import argparse
import os
sys.path.insert( 1, '../func' )
import consts as c
from copy import deepcopy

global img, image

pointSize = 4
dragInfo = { 'x': 0, 'y': 0, 'p': None, }

parser = argparse.ArgumentParser()
parser.add_argument( "video", help="path to folder with your video relative to /data/videos." )
parser.add_argument( "path", help="path to folder with your .p files relative to /data/images." )
args = parser.parse_known_args()[ 0 ]
args.path = "../../data/images/" + args.path
args.video = "../../data/videos/" + args.video + "/"


def readSkels():
    return pickle.load( open( args.path, "rb" ) ), int( args.path.split( '/' )[ -1 ].split( 'at' )[ 1 ].split( ".p" )[ 0 ].split( '_f' )[ 0 ] )


def getImagesNames():
    return [ name for name in os.listdir( args.video ) if name[ -5 ] == 'c' ]


def showImage( reset=False ):
    global img, image
    img = ImageTk.PhotoImage( Image.open( args.video + frames[ frameNumber ] ) )
    canvas.itemconfig( image, image=img )
    idx = frameNumber - beginFrame
    if idx < 0:
        return True
    if reset:
        tab = keypoints
    else:
        tab = newKeypoints
    allKeypoints = True
    for j, kp in enumerate( tab[ idx ] ):
        if kp[ 3 ] > 0:
            canvas.coords( points[ j ], kp[ 0 ] - pointSize, kp[ 1 ] - pointSize, kp[ 0 ] + pointSize, kp[ 1 ] + pointSize )
            canvas.itemconfig( points[ j ], fill="green" )
        elif kp[ 3 ] == -1:
            canvas.coords( points[ j ], kp[ 0 ] - pointSize, kp[ 1 ] - pointSize, kp[ 0 ] + pointSize, kp[ 1 ] + pointSize )
            canvas.itemconfig( points[ j ], fill="purple" )
            allKeypoints = False
        else:
            canvas.itemconfig( points[ j ], fill="red" )
            infoLabel.configure( text="Not detected!" )
            allKeypoints = False
    return allKeypoints


def firstImage():
    global img, image, points
    img = ImageTk.PhotoImage( Image.open( args.video + frames[ frameNumber ] ) )
    image = canvas.create_image( (0, 0), image=img, anchor='nw' )
    idx = frameNumber - beginFrame
    if idx < 0:
        return
    for kp in newKeypoints[ idx ]:
        if kp[ 3 ] > 0:
            points.append( canvas.create_oval( kp[ 0 ] - pointSize, kp[ 1 ] - pointSize, kp[ 0 ] + pointSize,
                                               kp[ 1 ] + pointSize, fill="green", outline="", tags=('pt',) ) )
        else:
            points.append( canvas.create_oval( 320 - pointSize, 5 - pointSize, 320 + pointSize,
                                               5 + pointSize, fill="red", outline="", tags=('pt',) ) )


def onPointPressed( event ):
    global dragInfo
    dragInfo[ 'x' ] = event.x
    dragInfo[ 'y' ] = event.y
    dragInfo[ 'p' ] = canvas.find_closest( event.x, event.y )[ 0 ]
    try:
        infoLabel.configure( text=c.keypoints[ points.index( dragInfo[ 'p' ] ) ] )
        canvas.itemconfig( dragInfo[ 'p' ], fill="purple" )
    except ValueError:
        pass


def onPointReleased( event ):
    dragInfo[ 'x' ] = 0
    dragInfo[ 'y' ] = 0
    idx = frameNumber - beginFrame
    coords = canvas.coords( dragInfo[ 'p' ] )
    x = int( coords[ 0 ] + pointSize )
    y = int( coords[ 1 ] + pointSize )
    try:
        i = points.index( dragInfo[ 'p' ] )
        newKeypoints[ idx ][ i ] = [ x, y, newKeypoints[ idx ][ i ][ 2 ], -1 ]
    except ValueError:
        pass
    dragInfo[ 'p' ] = None


def onPointMove( event ):
    if dragInfo[ 'p' ] is not None and dragInfo[ 'p' ] != image:
        x = event.x - dragInfo[ 'x' ]
        y = event.y - dragInfo[ 'y' ]
        canvas.move( dragInfo[ 'p' ], x, y )
        dragInfo[ 'x' ] = event.x
        dragInfo[ 'y' ] = event.y


def nextFrame():
    global frameNumber
    frameNumber = frameNumber + 1
    try:
        showImage()
    except IndexError:
        frameNumber = frameNumber - 1
        return


def skip():
    global frameNumber
    frameNumber = frameNumber + 1
    try:
        while showImage():
            frameNumber = frameNumber + 1
            pass
    except IndexError:
        frameNumber = frameNumber - 1
        return


def prevFrame():
    global frameNumber
    if frameNumber <= 0:
        return
    frameNumber = frameNumber - 1
    showImage()


def save():
    for frame in newKeypoints:
        for kp, _ in enumerate( frame ):
            if frame[ kp ][ 3 ] == -1:
                frame[ kp ][ 3 ] = 1.0
    pickle.dump( newKeypoints, open( f"{ args.path[ :-2 ] }_f.p", "wb" ) )


def resetKeys():
    showImage( True )
    idx = frameNumber - beginFrame
    if idx >= 0:
        newKeypoints[ idx ] = deepcopy( keypoints[ idx ] )


root = Tk()
root.title( "Keypoints editor" )

prevButton = Button( root, text="Previous", command=prevFrame, height=2, width=10 )
nextButton = Button( root, text="Next", command=nextFrame, height=2, width=10 )
resetButton = Button( root, text="Reset", command=resetKeys, height=2, width=10 )
saveButton = Button( root, text="Save", command=save, height=2, width=10 )
skipButton = Button( root, text="Skip", command=skip, height=2, width=10 )
preskipButton = Button( root, text="Preskip", height=2, width=10 )
canvas = Canvas( root, width=640, height=480 )
canvas.grid( row=0, column=0, columnspan=5 )
canvas.tag_bind( 'pt', '<ButtonPress-1>', onPointPressed )
canvas.tag_bind( 'pt', '<ButtonRelease-1>', onPointReleased )
canvas.tag_bind( 'pt', '<B1-Motion>', onPointMove )
nextButton.grid( row=1, column=3 )
prevButton.grid( row=1, column=2 )
skipButton.grid( row=1, column=4 )
preskipButton.grid( row=1, column=1 )
resetButton.grid( row=2, column=3 )
saveButton.grid( row=2, column=4 )

infoLabel = Label( root, text="Keypoint", font=( "", 12 ), width=30, anchor="w" )
infoLabel.grid( row=1, column=0 )

frames = getImagesNames()
keypoints, beginFrame = readSkels()
newKeypoints = deepcopy( keypoints )
frameNumber = 0
points = []
firstImage()

root.mainloop()
