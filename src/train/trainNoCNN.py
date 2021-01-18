import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt

# os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
import tensorflow as tf

sys.path.insert( 1, '../func' )
import model
import consts as c


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument( "train_ds", help="Name or path to your dataset relative to /data/datasets." )
    parser.add_argument( "-v", "--validation_ds", help="Path to validation dataset relative to /data/datasets." )
    parser.add_argument( "-m", "--model_name", help="Name of model you want to continue training. If none, new model will be created." )
    parser.add_argument( "-o", "--output_model", help="Name of output model. If none, model won't be saved." )
    parser.add_argument( "-z", "--no_z", action="store_true" )
    return parser.parse_known_args()[ 0 ]


def readDataset( dsName ):
    with np.load( "../../data/datasets/" + dsName, allow_pickle=True ) as data:
        img = data[ 'images' ]
        lab = data[ 'labels' ]
    print( "Dataset loaded." )
    if args.no_z:
        img[ :, :, :, 0 ] = np.zeros( img.shape[ :3 ] )
    return img, lab


def shuffleAndSplit( ds, datasetSize, trainSizeFactor ):
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
        mod = model.getModel( "smallVGG" )
    print()
    print( mod.summary() )
    return mod


def saveModel():
    if args.output_model is not None:
        m.save( '../../data/models/' + args.output_model )
        print( "Model saved to /data/models/" + args.output_model )
    else:
        print( "Model not saved." )


def showPlots( hist ):
    accuracy = plt.figure( 0 )
    plt.plot( hist[ 'accuracy' ] )
    plt.plot( hist[ 'val_accuracy' ] )
    plt.title( 'Model accuracy' )
    plt.ylabel( 'accuracy' )
    plt.xlabel( 'epoch' )
    plt.legend( [ 'train', 'val' ], loc='upper left' )
    loss = plt.figure( 1 )
    plt.plot( hist[ 'loss' ] )
    plt.plot( hist[ 'val_loss' ] )
    plt.title( 'Model loss' )
    plt.ylabel( 'loss' )
    plt.xlabel( 'epoch' )
    plt.legend( [ 'train', 'val' ], loc='upper left' )
    plt.show()


if __name__ == '__main__':
    args = parseArgs()
    images, labels = readDataset( args.train_ds )
    images = images / 255.0
    dataset = tf.data.Dataset.from_tensor_slices( ( images, labels ) )
    if args.validation_ds is not None:
        train, _ = shuffleAndSplit( dataset, len( images ), 1.0 )
        images, labels = readDataset( args.validation_ds )
        images = images / 255.0
        dataset = tf.data.Dataset.from_tensor_slices( ( images, labels ) )
        validation, _ = shuffleAndSplit( dataset, len( images ), 1.0 )
    else:
        train, validation = shuffleAndSplit( dataset, len( images ), 0.8 )

    m = getModel()

    train = train.batch( c.batchSize )
    validation = validation.batch( c.batchSize )
    history = m.fit( train, epochs=3, batch_size=c.batchSize, validation_data=validation, initial_epoch=0 )
    # testLoss, testAccuracy = m.evaluate( testDataset )
    # print( "Test loss = " + str( testLoss ) + "\nTest accuracy = " + str( testAccuracy ) )
    saveModel()
    showPlots( history.history )
