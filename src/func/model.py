from tensorflow.keras import layers, models
import consts as cs


# Function creates and returns CNN model based on VGG network
def getModel():
    model = models.Sequential()
    # first group
    model.add( layers.Conv2D( filters = 64, kernel_size = 3, strides = 1, padding = 'same',
                              activation = 'relu', input_shape = (cs.framesNumber, cs.keypointsNumber, 3) ) )
    model.add( layers.Conv2D( filters = 64, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu' ) )
    model.add( layers.MaxPooling2D( pool_size = 2, strides = 1, padding = 'same' ) )

    # second group
    model.add( layers.Conv2D( filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu' ) )
    model.add( layers.Conv2D( filters = 128, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu' ) )
    model.add( layers.MaxPooling2D( pool_size = 2, padding = 'same' ) )

    # third group
    model.add( layers.Conv2D( filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu' ) )
    model.add( layers.Conv2D( filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation = 'relu' ) )
    model.add( layers.GlobalAveragePooling2D() )

    # end - now we have vector
    model.add( layers.Dense( cs.posesNumber, activation = 'softmax' ) )
    model.compile( optimizer = 'adam',
                   loss = 'sparse_categorical_crossentropy',
                   metrics = [ 'accuracy' ] )
    return model
