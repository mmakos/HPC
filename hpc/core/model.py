import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import layers, models, applications, optimizers
import hpc.consts as c


# Function creates and returns CNN model based on VGG network
def getModel(m="smallVGG"):
    if m == "smallVGG":
        return getSmallVGG()
    elif m == "MobileNet":
        return getMobileNet()
    elif m == "VGG16":
        return getVGG16()
    elif m == "noCNN":
        return getNoCNN()


def getSmallVGG():
    model = models.Sequential()
    # first group
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                            activation='relu', input_shape=(c.keypointsNumber, c.framesNumber, 3)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1, padding='same'))

    # second group
    model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, padding='same'))

    # third group
    model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.GlobalAveragePooling2D())

    # end - now we have vector
    model.add(layers.Dense(8, activation='softmax'))
    model.compile(optimizer=optimizers.Adam(learning_rate=c.learningRate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def getMobileNet():
    return applications.MobileNetV3Small(input_shape=(c.keypointsNumber, c.framesNumber, 3))


def getVGG16():
    model = models.Sequential()
    # first group
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same',
                            activation='relu', input_shape=(c.keypointsNumber, c.framesNumber, 3)))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # second group
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # third group
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # third group
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    # third group
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=2, strides=1))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=4096, activation='relu'))
    model.add(layers.Dense(units=4096, activation='relu'))

    # end - now we have vector
    model.add(layers.Dense(len(c.poses), activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def getNoCNN():
    model = models.Sequential([
        layers.Flatten(input_shape=(c.keypointsNumber, c.framesNumber, 3)),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(c.poses), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
