import argparse
import os
import pickle
from tkinter import *

from PIL import Image, ImageTk

import hpc.consts as c
from copy import deepcopy

global img, image

pointSize = 4
dragInfo = {'x': 0, 'y': 0, 'p': None, }

parser = argparse.ArgumentParser()
parser.add_argument("video", help="path to folder with your video relative to /data/videos.")
parser.add_argument("path", help="path to your .p file files relative to /data/images.")
args = parser.parse_known_args()[0]
args.path = "data/images/" + args.path
args.video = "data/videos/" + args.video + "/"


def readSkels():
    return pickle.load(open(args.path, "rb")), int(args.path.split('/')[-1].split('at')[1].split(".p")[0].split('_f')[0])


def getImagesNames():
    return [name for name in os.listdir(args.video) if name[-5] == 'c']


def showImage(reset=False):
    global img, image
    img = ImageTk.PhotoImage(Image.open(args.video + frames[frameNumber]))
    canvas.itemconfig(image, image=img)
    frameLabel.configure(text=f"Frame {frameNumber}")
    idx = frameNumber - beginFrame
    if idx < 0:
        return True
    if reset:
        tab = keypoints
    else:
        tab = newKeypoints
    allKeypoints = True
    if len(points) == 0:
        firstImage()
        return True
    drawLines(tab[idx])
    for j, kp in enumerate(tab[idx]):
        if kp[3] > 0:
            canvas.coords(points[j], kp[0] - pointSize, kp[1] - pointSize, kp[0] + pointSize, kp[1] + pointSize)
            canvas.itemconfig(points[j], fill="green")
        elif kp[3] == -1:
            canvas.coords(points[j], kp[0] - pointSize, kp[1] - pointSize, kp[0] + pointSize, kp[1] + pointSize)
            canvas.itemconfig(points[j], fill="purple")
            allKeypoints = False
        else:
            canvas.itemconfig(points[j], fill="red")
            infoLabel.configure(text="Skeleton incomplete!")
            allKeypoints = False
    if allKeypoints:
        infoLabel.configure(text="Skeleton complete.")
    return allKeypoints


def drawLines(kps, first=False):
    done = []
    for ki, k in enumerate(c.connections):
        temp = []
        if first:
            for kj in k:  # for every neighbor of proceeded keypoint
                if kj not in done:
                    if ki in [1, 2, 3, 12, 13, 14]:
                        color = "yellow"
                    else:
                        color = "blue"
                    temp.append(canvas.create_line(kps[ki][0], kps[ki][1], kps[kj][0], kps[kj][1], width=2, fill=color))
                else:
                    temp.append(lines[kj][c.connections[kj].index(ki)])
            lines.append(temp)
        else:
            for kji, kj in enumerate(k):
                if kj not in done:
                    if (kps[ki][0] > 0 or kps[ki][1] > 0) and (kps[kj][0] > 0 or kps[kj][1] > 0):
                        canvas.coords(lines[ki][kji], kps[ki][0], kps[ki][1], kps[kj][0], kps[kj][1])
        done.append(ki)


def reverse():
    idx = frameNumber - beginFrame
    if idx < 0:
        return
    kps = newKeypoints[idx]
    kps[2], kps[5] = kps[5], kps[2]
    kps[3], kps[6] = kps[6], kps[3]
    kps[4], kps[7] = kps[7], kps[4]
    kps[9], kps[12] = kps[12], kps[9]
    kps[10], kps[13] = kps[13], kps[10]
    kps[11], kps[14] = kps[14], kps[11]
    showImage()


def firstImage():
    global img, image, points
    img = ImageTk.PhotoImage(Image.open(args.video + frames[frameNumber]))
    image = canvas.create_image((0, 0), image=img, anchor='nw')
    idx = frameNumber - beginFrame
    if idx < 0:
        return
    drawLines(newKeypoints[idx], True)
    for kp in newKeypoints[idx]:
        if kp[3] > 0:
            points.append(canvas.create_oval(kp[0] - pointSize, kp[1] - pointSize, kp[0] + pointSize,
                                             kp[1] + pointSize, fill="green", outline="", tags=('pt',)))
        else:
            points.append(canvas.create_oval(320 - pointSize, 5 - pointSize, 320 + pointSize,
                                             5 + pointSize, fill="red", outline="", tags=('pt',)))


def onPointPressed(event):
    global dragInfo
    dragInfo['x'] = event.x
    dragInfo['y'] = event.y
    dragInfo['p'] = canvas.find_closest(event.x, event.y)[0]
    try:
        infoLabel.configure(text=c.keypoints[points.index(dragInfo['p'])])
        canvas.itemconfig(dragInfo['p'], fill="purple")
    except ValueError:
        pass


def onPointReleased(event):
    dragInfo['x'] = 0
    dragInfo['y'] = 0
    idx = frameNumber - beginFrame
    coords = canvas.coords(dragInfo['p'])
    x = int(coords[0] + pointSize)
    y = int(coords[1] + pointSize)
    try:
        i = points.index(dragInfo['p'])
        newKeypoints[idx][i] = [x, y, newKeypoints[idx][i][2], -1]
        drawLines(newKeypoints[idx])
    except ValueError:
        pass
    dragInfo['p'] = None


def onPointMove(event):
    if dragInfo['p'] is not None and dragInfo['p'] != image:
        x = event.x - dragInfo['x']
        y = event.y - dragInfo['y']
        canvas.move(dragInfo['p'], x, y)
        dragInfo['x'] = event.x
        dragInfo['y'] = event.y


def takeFromPrevious():
    idx = frameNumber - beginFrame
    if idx <= 0:
        return
    for kpi, kp in enumerate(newKeypoints[idx]):
        if kp[0] == 0 and kp[1] == 0 and (newKeypoints[idx - 1][kpi][0] != 0 or newKeypoints[idx - 1][kpi][1] != 0):
            newKeypoints[idx][kpi] = newKeypoints[idx - 1][kpi]
    showImage()


def nextFrame(event=None):
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


def prevFrame(event=None):
    global frameNumber
    if frameNumber <= 0:
        return
    frameNumber = frameNumber - 1
    showImage()


def save():
    for frame in newKeypoints:
        for kp, _ in enumerate(frame):
            if frame[kp][3] == -1:
                frame[kp][3] = 1.0
    pickle.dump(newKeypoints, open(f"{args.path[:-2]}_f.p", "wb"))
    infoLabel.configure(text="Saved.")


def resetKeys():
    showImage(True)
    idx = frameNumber - beginFrame
    if idx >= 0:
        newKeypoints[idx] = deepcopy(keypoints[idx])


root = Tk()
root.title("Keypoints editor")

prevButton = Button(root, text="Previous", command=prevFrame, height=2, width=10)
root.bind("<Left>", prevFrame)
nextButton = Button(root, text="Next", command=nextFrame, height=2, width=10)
root.bind("<Right>", nextFrame)
resetButton = Button(root, text="Reset", command=resetKeys, height=2, width=10)
takePrevButton = Button(root, text="Take prev", command=takeFromPrevious, height=2, width=10)
saveButton = Button(root, text="Save", command=save, height=2, width=10)
skipButton = Button(root, text="Skip", command=skip, height=2, width=10)
reverseButton = Button(root, text="Reverse", command=reverse, height=2, width=10)
canvas = Canvas(root, width=640, height=480)
canvas.grid(row=0, column=0, columnspan=5)
canvas.tag_bind('pt', '<ButtonPress-1>', onPointPressed)
canvas.tag_bind('pt', '<ButtonRelease-1>', onPointReleased)
canvas.tag_bind('pt', '<B1-Motion>', onPointMove)
nextButton.grid(row=1, column=3)
prevButton.grid(row=1, column=2)
skipButton.grid(row=1, column=4)
reverseButton.grid(row=1, column=1)
takePrevButton.grid(row=2, column=2)
resetButton.grid(row=2, column=3)
saveButton.grid(row=2, column=4)

infoLabel = Label(root, text="Keypoint", font=("", 12), width=30, anchor="w")
infoLabel.grid(row=1, column=0)
frameLabel = Label(root, text="Frame ", font=("", 12), width=30, anchor="w")
frameLabel.grid(row=2, column=0)

frames = getImagesNames()
keypoints, beginFrame = readSkels()
newKeypoints = deepcopy(keypoints)
frameNumber = 0
points = []
lines = []
firstImage()

root.mainloop()
