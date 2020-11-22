import sys
import argparse
import numpy as np
import os

# os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
import tensorflow as tf

sys.path.insert( 1, '../func' )
import model
import consts as c


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "dataset_name", help="Name or path to your dataset relative to /data/datasets." )
    parser.add_argument( "-m", "--model_name", help="Name of model you want to continue training. If none, new model will be created." )
    parser.add_argument( "-o", "--output_model", help="Name of output model. If none, model won't be saved." )
    return parser.parse_known_args()[ 0 ]


def readDataset():
    with np.load( "../../data/datasets/" + args.dataset_name, allow_pickle=True ) as data:
        img = data[ 'images' ]
        lab = data[ 'labels' ]
    print( "Dataset loaded." )
    return img, lab


def getTrainTest( ds, datasetSize, trainSizeFactor ):
    if not 0 < trainSizeFactor <= 1:
        raise ValueError( "Train size factor must be in <0, 1>" )
    ds = ds.shuffle( datasetSize )
    return ds.take( int( trainSizeFactor * datasetSize ) ), ds.skip( int( trainSizeFactor * datasetSize ) )


def getModel():
    try:
        mod = tf.keras.models.load_model( '../../data/models/' + args.model_name )
        print( "Model " + args.model_name + " loaded." )
    except Exception:
        print( "Creating new model." )
        mod = model.getModel()
    print()
    print( mod.summary() )
    return mod


def saveModel():
    if args.output_model is not None:
        m.save( '../../data/models/' + args.output_model )
        print( "Model saved to /data/models/" + args.output_model )
    else:
        print( "Model not saved." )


if __name__ == '__main__':
    args = parseArgs()
    images, labels = readDataset()
    m = getModel()
    dataset = tf.data.Dataset.from_tensor_slices( ( images, labels ) )
    trainDataset, testDataset = getTrainTest( dataset, len( images ), 1.0 )

    print( "Train dataset: " + str( trainDataset ) )
    print( "Test dataset: " + str( testDataset ) )

    trainDataset = trainDataset.batch( c.batchSize )
    testDataset = testDataset.batch( c.batchSize )
    # trainDataset.reshape( -1, c.keypointsNumber, c.framesNumber, 3 )
    m.fit( trainDataset, epochs=3, batch_size=c.batchSize )
    # testLoss, testAccuracy = m.evaluate( testDataset )
    # print( "Test loss = " + str( testLoss ) + "\nTest accuracy = " + str( testAccuracy ) )

    saveModel()
