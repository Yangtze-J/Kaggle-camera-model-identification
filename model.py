from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from config import *


def Xception():
    from keras.applications.xception import Xception
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D

    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_image_shape)
    base_model.summary()
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(2048, activation='relu')(x)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    for layer in base_model.layers:
        layer.trainable = True

    RMS = optimizers.RMSprop(lr=0.001, decay=1e-7)
    model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])
    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # for layer in model.layers[:249]:
    #     layer.trainable = False
    # for layer in model.layers[249:]:
    #     layer.trainable = True


def IceptionResnet_V2():
    from keras.applications.inception_resnet_v2 import InceptionResNetV2

    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_image_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(2048, activation='relu')(x)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    for layer in base_model.layers:
        layer.trainable = True
    RMS = optimizers.RMSprop(lr=0.001, decay=1e-7)
    model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])
    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def InceptionV3():
    from keras.applications.inception_v3 import InceptionV3

    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_image_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(2048, activation='relu')(x)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    for layer in base_model.layers:
        layer.trainable = True
    RMS = optimizers.RMSprop(lr=0.001, decay=1e-7)
    model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])
    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def ResNet50():
    from keras.applications.resnet50 import ResNet50

    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_image_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(2048, activation='relu')(x)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    for layer in base_model.layers:
        layer.trainable = True
    RMS = optimizers.RMSprop(lr=0.001, decay=1e-7)
    model.compile(optimizer=RMS, loss='categorical_crossentropy', metrics=['accuracy'])
    # compile the model (should be done *after* setting layers to non-trainable)
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def model_create():
    # ## Define a Convolutional Neural Network

    dr = 0.6
    seed = 7
    np.random.seed(seed)

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
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

    # You can view a summary of the network using the `summary()` function: