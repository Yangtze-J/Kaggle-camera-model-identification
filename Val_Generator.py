import os
import random
from PIL import Image
ROOT_DIR = os.getcwd()
DEFAULT_TRAIN_PATH = os.path.join(ROOT_DIR, "train")
DEFAULT_VAL_PATH = os.path.join(ROOT_DIR, "val")
DEFAULT_NJPG_PATH = os.path.join(ROOT_DIR, "NJPG")


label_list = sorted(os.listdir(DEFAULT_TRAIN_PATH), reverse=False)
print("Class list:", label_list)


def val_generator():
    # Make val dir
    if os.path.exists(DEFAULT_VAL_PATH) is False:
        os.makedirs(DEFAULT_VAL_PATH)
    # Make sub dir
    for cls_name in label_list:
        if os.path.exists(DEFAULT_VAL_PATH + '/' + cls_name) is False:
            os.makedirs(DEFAULT_VAL_PATH + '/' + cls_name)
            print("Make dir: ", cls_name)

    for i, cls in enumerate(label_list):
        image_name_list = sorted(os.listdir(DEFAULT_TRAIN_PATH+'/'+cls), reverse=False)
        print("Working on: {0}, {1} images in train class.".format(cls, len(image_name_list)))

        for _ in range(int(len(image_name_list)/10)):
            choice = random.choice(image_name_list)
            os.rename(DEFAULT_TRAIN_PATH+'/'+cls+'/'+choice, DEFAULT_VAL_PATH+'/'+cls+'/'+choice)
            print("Moved:", cls + '/' + choice)
            image_name_list.remove(choice)


def val_back_train():
    for i, cls in enumerate(label_list):
        image_name_list = sorted(os.listdir(DEFAULT_VAL_PATH+'/'+cls), reverse=False)
        print("Working on: {0}, {1} images in val class.".format(cls, len(image_name_list)))
        for choice in image_name_list:
            os.rename(DEFAULT_VAL_PATH + '/' + cls + '/' + choice, DEFAULT_TRAIN_PATH + '/' + cls + '/' + choice)
            print("Moved:", cls + '/' + choice)


def clean_train_for_jpg():
    # Make non jpg dir
    if os.path.exists(DEFAULT_NJPG_PATH) is False:
        os.makedirs(DEFAULT_NJPG_PATH)
    # Make sub dir
    for cls_name in label_list:
        if os.path.exists(DEFAULT_NJPG_PATH + '/' + cls_name) is False:
            os.makedirs(DEFAULT_NJPG_PATH + '/' + cls_name)
            print("Make dir: ", cls_name)

    for i, cls in enumerate(label_list):
        image_name_list = sorted(os.listdir(DEFAULT_TRAIN_PATH+'/'+cls), reverse=False)
        print("Working on: {0}, {1} images in val class.".format(cls, len(image_name_list)))
        for choice in image_name_list:
            try:
                with Image.open(DEFAULT_TRAIN_PATH+'/'+cls+'/'+choice) as opened_image:
                    if opened_image.format is not 'JPEG':
                        # Delete not JPEG image
                        os.rename(DEFAULT_TRAIN_PATH+'/'+cls+'/'+choice,
                                  DEFAULT_NJPG_PATH + '/' + cls + '/' + choice)
                        print("Moved:", cls + '/' + choice)
                        image_name_list.remove(choice)
            except:
                print("{0} is not image".format(choice))


if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="'generate' or 'back' on Dataset")
    parser.add_argument("command",
                        metavar="<command>",
                        help="'generate' or 'back' on Dataset")

    args = parser.parse_args()
    print("Command: ", args.command)
    if args.command == "generate":
        val_generator()
    elif args.command == "back":
        val_back_train()
    elif args.command == "clean":
        clean_train_for_jpg()
    else:
        print("Command error, 'generate' or 'back'")

