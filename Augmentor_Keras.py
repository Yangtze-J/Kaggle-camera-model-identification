
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import numpy as np


# ## Define a Convolutional Neural Network
# 
# Once the libraries have been imported, we define a small convolutional neural network. See the Keras documentation for details of this network: <https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py> 
# 
# It is a three layer deep neural network, consisting of 2 convolutional layers and a fully connected layer:

# In[19]:


num_classes = 10
input_shape = (64, 64, 3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
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

# In[3]:


# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])


# You can view a summary of the network using the `summary()` function:

# In[20]:


model.summary()



# In[2]:


p = Augmentor.Pipeline("/home/zhangbin/Image_Recog/train")


# ## Add Operations to the Pipeline
# 
# Now that a pipeline object `p` has been created, we can add operations to the pipeline. Below we add several simple  operations:

# In[4]:


p.rotate90(probability=0.5)
p.flip_top_bottom(probability=0.5)
p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
p.crop_by_size(probability=1, width=64, height=64, centre=False)


# You can view the status of pipeline using the `status()` function, which shows information regarding the number of classes in the pipeline, the number of images, and what operations have been added to the pipeline:

# In[5]:


p.status()


# ## Creating a Generator
# 
# A generator will create images indefinitely, and we can use this generator as input into the model created above. The generator is created with a user-defined batch size, which we define here in a variable named `batch_size`. This is used later to define number of steps per epoch, so it is best to keep it stored as a variable.

# In[6]:


batch_size = 32
g = p.keras_generator(batch_size=batch_size)


# The generator can now be used to created augmented data. In Python, generators are invoked using the `next()` function - the Augmentor generators will return images indefinitely, and so `next()` can be called as often as required. 
# 
# You can view the output of generator manually:

# In[7]:


images, labels = next(g)


# Images, and their labels, are returned in batches of the size defined above by `batch_size`. The `image_batch` variable is a tuple, containing the augmentented images and their corresponding labels.
# 
# To see the label of the first image returned by the generator you can use the array's index:

# In[8]:


print(labels[0])


# In[12]:


print(images.shape)


# ## Train the Network
# 
# We train the network by passing the generator, `g`, to the model's fit function. In Keras, if a generator is used we used the `fit_generator()` function as opposed to the standard `fit()` function. Also, the steps per epoch should roughly equal the total number of images in your dataset divided by the `batch_size`.
# 
# Training the network over 5 epochs, we get the following output:

# In[ ]:


ite = 200

for iteration in range(1, ite):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    h = model.fit_generator(g, steps_per_epoch=len(p.augmentor_images) / batch_size, epochs=5, verbose=1)

    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    if iteration % 10 == 0:
        # if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
        #     os.makedirs(DEFAULT_WEIGHT_PATH)
        model.save('/my_model.h5')
        print('saved model')
    else:
        if iteration == ite - 1:
            # if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
            #     os.makedirs(DEFAULT_WEIGHT_PATH)
            model.save('/my_model.h5')
            print('saved model')

# ## Summary
# 
# Using Augmentor with Keras means only that you need to create a generator when you are finished creating your pipeline. This has the advantage that no images need to be saved to disk and are augmented on the fly.
