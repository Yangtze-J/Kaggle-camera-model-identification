import os
import Augmentor
import os
import csv
import random
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
import numpy as np


# Root directory of the project
ROOT_DIR = os.getcwd()
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "model")
DEFAULT_TRAIN_PATH = os.path.join(ROOT_DIR, "train")
DEFAULT_TEST_PATH = os.path.join(ROOT_DIR, "test")
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "log")
DEFAULT_VAL_PATH = os.path.join(ROOT_DIR, "val")

input_image_shape = (256, 256, 3)

train_batch_size = 64
val_batch_size = 64

evaluate_size = 100
pred_num_per_img = 10

num_classes = 10

label_list = sorted(os.listdir(DEFAULT_TRAIN_PATH), reverse=False)
