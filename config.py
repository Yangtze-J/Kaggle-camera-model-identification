import os
import argparse
import re
import csv
import random
from PIL import Image
import keras
from keras import backend as K
from keras.models import load_model
import numpy as np
from keras.utils import multi_gpu_model



# Root directory of the project
ROOT_DIR = os.getcwd()
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "model")
DEFAULT_TRAIN_PATH = os.path.join(ROOT_DIR, "train")
DEFAULT_TEST_PATH = os.path.join(ROOT_DIR, "test")
DEFAULT_LOG_PATH = os.path.join(ROOT_DIR, "log")
DEFAULT_VAL_PATH = os.path.join(ROOT_DIR, "val")


train_batch_size = 48
val_batch_size = 48

evaluate_size = 100
pred_num_per_img = 10

num_classes = 10

label_list = sorted(os.listdir(DEFAULT_TRAIN_PATH), reverse=False)
