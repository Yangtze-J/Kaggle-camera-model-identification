
# coding: utf-8

# # Training a Neural Network using Augmentor and Keras
# 
# In this notebook, we will train a simple convolutional neural network on the MNIST dataset using Augmentor to augment images on the fly using a generator.
# 
# ## Import Required Libraries
# 
# We start by making a number of imports:

# In[1]:


import Augmentor
import keras
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np


# Root directory of the project
ROOT_DIR = os.getcwd()
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "model")
DEFAULT_TRAIN_PATH = os.path.join(ROOT_DIR, "train")
input_image_shape = (64, 64, 3)
batch_size = 32
evaluate_size = 100


def model_create():
    # ## Define a Convolutional Neural Network

    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    # Once a network has been defined, you can compile it so that the model is ready to be trained with data:

    # model.compile(loss=keras.losses.categorical_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])

    # You can view a summary of the network using the `summary()` function:
    model.summary()
    return model


def train(model=None, ite=200):

    if model is None:
        model = model_create()
    else:
        model = load_model(model)

    p = Augmentor.Pipeline(DEFAULT_TRAIN_PATH)
    # ## Add Operations to the Pipeline
    #
    # Now that a pipeline object `p` has been created,
    # we can add operations to the pipeline.
    # Below we add several simple  operations:
    width = input_image_shape[0]
    height = input_image_shape[1]

    p.rotate90(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
    p.crop_by_size(probability=1, width=width, height=height, centre=False)

    # You can view the status of pipeline using the `status()` function,
    # which shows information regarding the number of classes in the pipeline,
    # the number of images, and what operations have been added to the pipeline:

    p.status()

    # ## Creating a Generator
    #
    # A generator will create images indefinitely,
    # and we can use this generator as input into the model created above.
    # The generator is created with a user-defined batch size,
    # which we define here in a variable named `batch_size`.
    # This is used later to define number of steps per epoch,
    # so it is best to keep it stored as a variable.

    g = p.keras_generator(batch_size=batch_size)

    # The generator can now be used to created augmented data.
    # In Python, generators are invoked using the `next()` function -
    # the Augmentor generators will return images indefinitely,
    # and so `next()` can be called as often as required.
    #
    # You can view the output of generator manually:

    images, labels = next(g)

    # Images, and their labels, are returned in batches of the size defined above by `batch_size`.
    # The `image_batch` variable is a tuple, containing the augmentented images and their corresponding labels.
    #m
    # To see the label of the first image returned by the generator you can use the array's index:

    print(labels[0])
    print(images.shape)

    # ## Train the Network
    #
    # We train the network by passing the generator,
    # `g`, to the model's fit function.
    # In Keras, if a generator is used we used the `fit_generator()` function
    # as opposed to the standard `fit()` function.
    #Also, the steps per epoch should roughly equal the total number of images
    # in your dataset divided by the `batch_size`.
    #
    # Training the network over 5 epochs, we get the following output:

    for iteration in range(1, ite):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        h = model.fit_generator(g, steps_per_epoch=len(p.augmentor_images) / batch_size, epochs=5, verbose=1)

        if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
            os.makedirs(DEFAULT_WEIGHT_PATH)
        model.save(DEFAULT_WEIGHT_PATH+"/my_model.h5")
        print('saved model')


def evaluate(model):
    model = load_model(model)
    p = Augmentor.Pipeline(DEFAULT_TRAIN_PATH)

    width = input_image_shape[0]
    height = input_image_shape[1]

    p.rotate90(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
    p.crop_by_size(probability=1, width=width, height=height, centre=False)

    p.status()

    g = p.keras_generator(batch_size=batch_size)
    images, labels = next(g)
    # x_eval, y_eval, _, _ = generate_data(EVAL_SIZE)
    loss, acc = model.evaluate(images, labels,
                               batch_size=evaluate_size)
    print("The loss is: {0:>10.5}\nThe accuracy is: {1:>10.5%}".format(loss, acc))


# ## Summary
# 
# Using Augmentor with Keras means only that you need to create a generator
# when you are finished creating your pipeline.
# This has the advantage that no images need to be saved to disk and are augmented on the fly.


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Image Rec.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test' on Image Rec")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/my_model.h5",
                        help="Path to my_model.h5 file")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    if args.command == "train":
        if args.model is None:
            train()
        else:
            train(args.model)
    elif args.command == "evaluate":
        assert args.model is not None, "Please load a model..."
        evaluate(args.model)
    elif args.command == "debug":
        model = model_create()
        model.save(DEFAULT_WEIGHT_PATH+"/my_model.h5")

	#     assert args.model is not None, "Please load a model..."
    #     test(args.model)
    # elif args.command == "plot":
    #     assert args.model is not None, "Please load a model..."
    #     plot(args.model)model.save(DEFAULT_WEIGHT_PATH+"/my_model.h5")

