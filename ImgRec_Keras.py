
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
import os
import csv
import random
from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# Root directory of the project
ROOT_DIR = os.getcwd()
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "model")
DEFAULT_TRAIN_PATH = os.path.join(ROOT_DIR, "train")
DEFAULT_TEST_PATH = os.path.join(ROOT_DIR, "test")
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "log")

input_image_shape = (128, 128, 3)
batch_size = 64
evaluate_size = 100
pred_num_per_img = 10
num_classes = 10

label_list = sorted(os.listdir(DEFAULT_TRAIN_PATH), reverse=False)


def fine_tune_model():

    from keras.applications.xception import Xception
    # from keras.optimizers import SGD
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D

    # create the base pre-trained model
    # base_model = InceptionV3(weights='imagenet', include_top=False)
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_image_shape)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have num_classes classes
    predictions = Dense(num_classes, activation='softmax')(x)
    #
    # # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional Xception layers
    for layer in base_model.layers:
        layer.trainable = False
    base_model.summary()

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model
    # train the model on the new data for a few epochs
    # model.fit_generator(...)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # for i, layer in enumerate(model.layers):
    #     print(i, layer.name)

    # # we chose to train the top 2 inception blocks, i.e. we will freeze
    # # the first 249 layers and unfreeze the rest:
    # for layer in model.layers[:249]:
    #     layer.trainable = False
    # for layer in model.layers[249:]:
    #     layer.trainable = True


def model_create():
    # ## Define a Convolutional Neural Network

    dr = 0.6
    seed = 7  
    np.random.seed(seed)  
    
    model = Sequential()  
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=input_image_shape,padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(Dropout(dr))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(Dropout(dr))
    model.add(MaxPooling2D(pool_size=(2,2)))  
    model.add(Flatten())  
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(dr))
    model.add(Dense(num_classes,activation='softmax'))  
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()    
    return model

    # You can view a summary of the network using the `summary()` function:


def train(model=None, personal_model=None, ite=200, changelr=None):

    if model is None:
        if personal_model is True:
            model = model_create()
        else:
            model = fine_tune_model()
    else:
        model = load_model(model)

    # Finish load model
    model.summary()

    p = Augmentor.Pipeline(DEFAULT_TRAIN_PATH)
    # ## Add Operations to the Pipeline
    #
    # Now that a pipeline object `p` has been created,
    # we can add operations to the pipeline.
    # Below we add several simple  operations:
    width = input_image_shape[0]
    height = input_image_shape[1]

    # p.rotate90(probability=0.5)
    # p.flip_top_bottom(probability=0.5)
    # p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
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

    len(p.augmentor_images)

    for iteration in range(1, ite):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        # steps_per_epoch=len(p.augmentor_images) / batch_size
        h = model.fit_generator(g, steps_per_epoch=10, epochs=1, verbose=1)
        print('Model learning rate :', K.get_value(model.optimizer.lr))
        acc = h.history['acc']
        loss = h.history['loss']
        if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
            os.makedirs(DEFAULT_WEIGHT_PATH)
        model.save(DEFAULT_WEIGHT_PATH+"/my_new_model.h5")
        print("Iteration{0}: ,saved model".format(iteration))
        log_results('bin_', acc, loss)


def debug(model):
    model = load_model(model)
    print(K.get_value(model.optimizer.lr))


def log_results(filename, acc_log, loss_log):
    print("Saving log")
    if os.path.exists(DEFAULT_LOG_PATH) is False:
        os.makedirs(DEFAULT_LOG_PATH)
    # Save the results to a file so we can graph it later.
    with open(DEFAULT_LOG_PATH + '/' + filename + 'acc.csv', 'a', newline='') as data_dump:
        wr = csv.writer(data_dump)
        for acc_item in acc_log:
            wr.writerow([acc_item])

    with open(DEFAULT_LOG_PATH + '/' + filename + 'loss.csv', 'a', newline='') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow([loss_item])


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
    # a = images[0]
    # img = Image.fromarray(images[0]*255, 'RGB')
    # img.show()
    print(np.amax(images))
    loss, acc = model.evaluate(images, labels,
                               batch_size=evaluate_size)
    print("The loss is: {0:>10.5}\nThe accuracy is: {1:>10.5%}".format(loss, acc))
    

def predict(model):
    model = load_model(model)
    img_name_list = os.listdir(DEFAULT_TEST_PATH)
    result = []
    name = []
    for i, img_name in enumerate(img_name_list):
        im = Image.open(DEFAULT_TEST_PATH + "/" + img_name)
        print("predict " + img_name)
        w, h = im.size
        width = input_image_shape[0]
        height = input_image_shape[1]

        # Zero samples list
        pred_img_list = []
        # Generate random samples from every test image.
        for _ in range(pred_num_per_img):
            x = random.randint(0, w - width - 1)
            y = random.randint(0, h - height - 1)
            img = im.crop((x, y, x+width, y+width))
            # img.show()
            imarray = np.array(img)
            pred_img_list.append(imarray)
        # Test samples and get the most frequent result as the best
        pred_img_list = np.asarray(pred_img_list)
        pred_img_list = pred_img_list.astype('float32')
        pred_img_list = pred_img_list/255
        pred = model.predict(x=pred_img_list, batch_size=pred_num_per_img, verbose=1)
        pred = np.argmax(np.bincount(np.argmax(pred, axis=1)))

        # Append result and image name
        result.append(label_list[pred])
        name.append(img_name)

    # Save csv file as a result.
    with open('result.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(['fname', 'camera'])
        for i in range(len(result)):
            spamwriter.writerow([name[i], result[i]])
    print("Finished")


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
    parser.add_argument('--pm', required=False,
                        metavar="Use personal model?",
                        help="\'True\' or \'False\'")
    parser.add_argument('--changelr', required=False,
                        metavar="Use personal model?",
                        help="\'True\' or \'False\'")
    # parser.add_argument('--finetune', required=False,
    #                     metavar="/path/to/my_model.h5",
    #                     help="Path to my_model.h5 file")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Personal Model:", args.pm)
    print("Change Learning Rate:", args.changelr)
    if args.command == "train":
        train(model=args.model, personal_model=args.pm, changelr=args.changelr)
    elif args.command == "evaluate":
        assert args.model is not None, "Please load a model..."
        evaluate(args.model)
    elif args.command == "predict":
        assert args.model is not None, "Please load a model..."
        predict(args.model)
    elif args.command == "debug":
        debug(args.model)

