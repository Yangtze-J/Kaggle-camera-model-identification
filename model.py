from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from config import *


def model_create():
    # ## Define a Convolutional Neural Network

    dr = 0.6
    seed = 7
    np.random.seed(seed)
    global input_image_shape
    model = Sequential()
    model.add(Conv2D(32, (5, 5), strides=(1, 1), input_shape=input_image_shape, padding='valid', activation='relu',
                     kernel_initializer='uniform'))
    model.add(Dropout(dr))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Dropout(dr))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(num_classes, activation='softmax'))
    # model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

    # You can view a summary of the network using the `summary()` function: