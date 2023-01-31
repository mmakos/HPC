import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics
import tensorflow as tf

import hpc.consts as c


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to test dataset relative to /data/datasets.")
    parser.add_argument("model", help="Path to model relative to /data/models.")
    parser.add_argument("-d", "--dynamic", help="Path to model for dynamic poses (if hybrid solution).")
    return parser.parse_known_args()[0]


def readDataset(dsName):
    with np.load("data/datasets/" + dsName, allow_pickle=True) as data:
        img = data['images']
        lab = data['labels']
    print("Dataset loaded.")
    return img, lab


def getModel(modelName):
    mod = tf.keras.models.load_model('data/models/' + modelName)
    print("Model " + modelName + " loaded.")
    return mod


# function returns sum of distances of particular point between all frames
def getPointsDistance(img, points=(8, 9, 10, 12, 13, 14)):
    moveSum = np.zeros(shape=3)
    for k in points:
        for f in range(1, 32):
            moveSum = np.add(moveSum, np.fabs(np.subtract(img[k, f], img[k, f - 1])))
    return moveSum[0] * c.xDistCoefficient + moveSum[1] * c.yDistCoefficient


def splitDataset(img, lab):
    stat, dyn, lStat, lDyn = [], [], [], []
    for i, im in enumerate(img):
        if getPointsDistance(im, c.distancePoints) * 255 < c.statDynThreshold:
            stat.append(im)
            lStat.append(lab[i])
        else:
            dyn.append(im)
            lDyn.append(lab[i])
    return np.array(stat), np.array(dyn), np.array(lStat), np.array(lDyn)


poses = ("stanie", "siedz.", "leż.", "poch.", "klęcz.", "chodz,", "skak.")
args = getArgs()
images, labels = readDataset(args.dataset)
images = images / 255.0
model = getModel(args.model)
dModel = getModel(args.dynamic) if args.dynamic is not None else None

print(model.summary())

if dModel is not None:
    images, dImages, labels, dLabels = splitDataset(images, labels)
    np.argmax(dModel.predict(dImages), axis=1)
    np.argmax(model.predict(images), axis=1)
    yPred = np.concatenate((np.argmax(dModel.predict(dImages), axis=1), np.argmax(model.predict(images), axis=1)))
    yTrue = np.concatenate((dLabels, labels))
else:
    yPred = np.argmax(model.predict(images), axis=1)
    yTrue = labels

_, weights = np.unique(yTrue, return_counts=True)

confusionMatrix = metrics.confusion_matrix(yTrue, yPred, normalize='true')
confusionVector = [confusionMatrix[i][i] for i in range(len(confusionMatrix)) if confusionMatrix[i][i] != 0]
absoluteAccuracy = np.mean(confusionVector)
relativeAccuracy = np.average(confusionVector, weights=weights)

print("Confusion matrix = \n", confusionMatrix)
print(f"Absolute accuracy = {absoluteAccuracy}\nRelative accuracy = {relativeAccuracy}")
plt.figure()
sns.heatmap(confusionMatrix * 100, xticklabels=poses, yticklabels=poses, annot=True, fmt='.1f', cmap="YlOrBr")
plt.xlabel("Zaklasyfikowane pozy")
plt.ylabel("Rzeczywiste pozy")
plt.title(f"Dokładność = {round(absoluteAccuracy * 100, 2)}%")
plt.show()
