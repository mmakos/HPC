import tensorflow as tf
import sys
sys.path.insert( 1, '../func' )
from frame import Frame
from preprocess import preprocess
import consts as c
import os
import numpy as np
import display
from time import time
dir_path = os.path.dirname( os.path.realpath( __file__ ) )
sys.path.append( dir_path + '/../../externals/openpose/build/python/openpose/Release' )
os.environ[ 'PATH' ] = os.environ[ 'PATH' ] + ';' + dir_path + '/../../externals/openpose/build/x64/Release;' + dir_path + '/../../externals/openpose/build/bin;'
import pyopenpose as op


class Wrapper:
    # model is name of model
    def __init__( self, model, gpuMode=False, opParams=None ):
        self.model = None
        self.dynModel = None
        self.opWrapper = None
        self.frameNumber = 0
        self.time = time()
        self.__initOpenPose( opParams )
        self.__getModel( model, gpuMode )
        self.frame = Frame( self.model, self.dynModel )

    def __getModel( self, model, gpuMode ):
        if not gpuMode:
            tf.config.list_physical_devices( 'GPU' )
            try:
                # Disable all GPUs
                tf.config.set_visible_devices( [ ], 'GPU' )
                visible_devices = tf.config.get_visible_devices()
                for device in visible_devices:
                    assert device.device_type != 'GPU'
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        if type( model ) is str or ( type( model ) is tuple and len( model ) == 1 ):   # one model
            modName = model
            dynName = None
        else:   # two models
            modName = model[ 0 ]
            dynName = model[ 1 ]

        self.model = tf.keras.models.load_model( '../../data/models/' + modName )
        print( "Model " + modName + " loaded." )
        print( self.model.summary() )
        if dynName is not None:
            self.model = tf.keras.models.load_model( '../../data/models/' + dynName )
            print( "Model " + dynName + " loaded." )
            print( self.dynModel.summary() )
        else:
            self.dynModel = None

    def __initOpenPose( self, opParams ):
        # starting OpenPose
        if not opParams:
            opParams = dict()
        opParams[ "model_folder" ] = "../../externals/openpose/models/"
        opParams[ "render_threshold" ] = c.keypointThreshold
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure( opParams )
        self.opWrapper.start()

    def proceed( self, frame, noDepth=False, noPose=False, noSkeleton=False, showTime=True ):
        frameRGB, frameD = frame
        if self.frameNumber == 0:
            try:
                c.frameHeight, c.frameWidth, _ = frameRGB.shape
                c.depthHeight, c.depthWidth = frameD.shape
            except:
                pass

        datum = op.Datum()
        datum.cvInputData = frameRGB
        self.opWrapper.emplaceAndPop( [ datum ] )

        # array of people with keypoints is in datum.poseKeypoints
        # getSkeletons gives for every human skeleton image
        if not noSkeleton:
            frameRGB = datum.cvOutputData  # image is frame with drawn skeleton

        humans = preprocess( datum.poseKeypoints, frameD, noDepth )
        # convert frame to skeleton image
        poses = []
        if not noPose:
            poses = self.frame.proceedFrame( humans )
        for j, human in enumerate( humans ):
            try:
                display.displayPose( frameRGB, human, str( poses[ j ][ 1 ] ) + ": " +
                                     c.poses[ np.argmax( poses[ j ][ 0 ] ) ] +
                                     f" - { int( np.max( poses[ j ][ 0 ] ) * 100 ) }%" )
            except:
                pass
        if showTime:
            display.displayFrameTime( frameRGB, time() - self.time )
            self.time = time()

        self.frameNumber = self.frameNumber + 1
        return frameRGB, poses, humans
