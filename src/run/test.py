import tensorflow as tf
import argparse
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "dataset", help="Path to test dataset relative to /data/datasets." )
    parser.add_argument( "model", help="Path to model relative to /data/models." )
    return parser.parse_known_args()[ 0 ]


def readDataset( dsName ):
    with np.load( "../../data/datasets/" + dsName, allow_pickle=True ) as data:
        img = data[ 'images' ]
        lab = data[ 'labels' ]
    print( "Dataset loaded." )
    return img, lab


def getModel( modelName ):
    mod = tf.keras.models.load_model( '../../data/models/' + modelName )
    print( "Model " + modelName + " loaded." )
    return mod


poses = ( "stand", "sit", "lie", "lean", "kneel",
          "walk", "jump" )
args = getArgs()
images, labels = readDataset( args.dataset )
images = images / 255.0
model = getModel( args.model )
yPred = np.argmax( model.predict( images ), axis=1 )
yTrue = labels

_, weights = np.unique( yTrue, return_counts=True )

confusionMatrix = metrics.confusion_matrix( yTrue, yPred, normalize='true' )
confusionVector = [ confusionMatrix[ i ][ i ] for i in range( len( confusionMatrix ) ) if confusionMatrix[ i ][ i ] != 0 ]
absoluteAccuracy = np.mean( confusionVector )
relativeAccuracy = np.average( confusionVector, weights=weights )

print( "Confusion matrix = \n", confusionMatrix )
print( f"Absolute accuracy = { absoluteAccuracy }\nRelative accuracy = { relativeAccuracy }" )
plt.figure()
sns.heatmap( confusionMatrix, xticklabels=poses, yticklabels=poses, annot=True, fmt='.2g')
plt.xlabel( "Predicted poses" )
plt.ylabel( "True poses" )
plt.title( "Jump and walk" )
plt.show()
