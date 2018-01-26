from config import *
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from model import model_create
import cv2
from io import BytesIO
import jpeg4py as jpeg


# 	Class index: 0 Class label: HTC-1-M7
# 	Class index: 1 Class label: LG-Nexus-5x
# 	Class index: 2 Class label: Motorola-Droid-Maxx
# 	Class index: 3 Class label: Motorola-Nexus-6
# 	Class index: 4 Class label: Motorola-X
# 	Class index: 5 Class label: Samsung-Galaxy-Note3
# 	Class index: 6 Class label: Samsung-Galaxy-S4
# 	Class index: 7 Class label: Sony-NEX-7
# 	Class index: 8 Class label: iPhone-4s
# 	Class index: 9 Class label: iPhone-6


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Image Rec.')
parser.add_argument("command",
                    metavar="<command>",
                    help="'train' or 'test' on Image Rec")
parser.add_argument('-m', '--model', required=False,
                    metavar="/path/to/my_model.h5",
                    help="Path to my_model.h5 file")
parser.add_argument('-cm', '--classifier', type=str, default='Xception', help='Base classifier model to use')
parser.add_argument('-pm', required=False,
                    metavar="Use personal model?",
                    help="\'True\' or \'False\'")
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-cs', '--crop-size', type=int, default=128, help='Crop size')
parser.add_argument('-me', '--max-epoch', type=int, default=200, help='Epoch to run')

args = parser.parse_args()

CROP_SIZE = args.crop_size
input_image_shape = (CROP_SIZE, CROP_SIZE, 3)


def train(model_path=None, personal_model=None):

    if model_path is None:
        if personal_model is True:
            model = model_create()
            model_name = "personal_model"
        else:
            classifier = globals()[args.classifier]

            base_model = classifier(
                include_top=False,
                weights='imagenet',
                input_shape=input_image_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # let's add a fully-connected layer
            x = Dense(2048, activation='relu')(x)
            # and a logistic layer -- let's say we have num_classes classes
            predictions = Dense(num_classes, activation='softmax')(x)
            # # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional Xception layers
            for layer in base_model.layers:
                layer.trainable = True
            print(args.classifier + " Model Created")
            model_name = args.classifier
        last_epoch = 0
    else:
        model = load_model(model_path, compile=False)
        match = re.search(r'(\D*)-epoch(\d+)-(\d+)-(\d+).h5', args.model)
        model_name = match.group(1)
        last_epoch = int(match.group(2))
        print("Model name:{0}, last epoch:{1}".format(model_name, last_epoch))
    if args.gpus >= 2:
        model = multi_gpu_model(model, gpus=args.gpus)

    adm = keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
    # Finish load model
    model.summary()

    p = Augmentor.Pipeline(DEFAULT_TRAIN_PATH)
    # clean not jpg image
    for augmentor_image in p.augmentor_images:
        with Image.open(augmentor_image.image_path) as opened_image:
            if opened_image.format is not 'JPEG':
                p.augmentor_images.remove(augmentor_image)

    width = input_image_shape[0]
    height = input_image_shape[1]

    # p.flip_top_bottom(probability=0.1)
    p.crop_by_size(probability=1, width=width, height=height, centre=False)

    p.status()

    pg = p.keras_generator(batch_size=train_batch_size)

    v = Augmentor.Pipeline(DEFAULT_VAL_PATH)
    # clean not jpg image
    for augmentor_image in v.augmentor_images:
        with Image.open(augmentor_image.image_path) as opened_image:
            if opened_image.format is not 'JPEG':
                v.augmentor_images.remove(augmentor_image)

    v.crop_by_size(probability=1, width=width, height=height, centre=False)
    # v.status()

    vg = v.keras_generator(batch_size=val_batch_size)

    # You can view the output of generator manually:
    # images, labels = next(g)

    # len(p.augmentor_images)

    print()
    print('-' * 50)
    h = model.fit_generator(generator=pg, steps_per_epoch=len(p.augmentor_images)/train_batch_size,
                            epochs=args.max_epoch, verbose=1,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                                                     verbose=1, mode='auto'),
                                       keras.callbacks.ModelCheckpoint(DEFAULT_WEIGHT_PATH+"/"+model_name+
                                                                       "-epoch"+"{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5",
                                                                       monitor='val_loss', verbose=1,
                                                                       save_best_only=True, save_weights_only=False,
                                                                       mode='auto', period=1)],
                            validation_data=vg, validation_steps=len(v.augmentor_images)/val_batch_size,
                            initial_epoch=last_epoch,)
    print('Model learning rate :', K.get_value(model.optimizer.lr))
    acc = h.history['acc']
    loss = h.history['loss']
    if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
        os.makedirs(DEFAULT_WEIGHT_PATH)
    # model.save(DEFAULT_WEIGHT_PATH+"/new_model.h5")
    log_results('bin_', acc, loss)


def debug(model_path):
    model = load_model(model_path)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
        layer.trainable = True
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    model.save(model_path)


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


def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    else:
        assert False
    return im_decoded


def evaluate(model_path):
    model = load_model(model_path)
    p = Augmentor.Pipeline(DEFAULT_TRAIN_PATH)

    width = input_image_shape[0]
    height = input_image_shape[1]

    p.flip_top_bottom(probability=0.5)
    p.crop_by_size(probability=1, width=width, height=height, centre=False)

    p.status()

    g = p.keras_generator(batch_size=train_batch_size)
    images, labels = next(g)
    # x_eval, y_eval, _, _ = generate_data(EVAL_SIZE)
    # a = images[0]
    # img = Image.fromarray(images[0]*255, 'RGB')
    # img.show()
    print(np.amax(images))
    loss, acc = model.evaluate(images, labels,
                               train_batch_size=evaluate_size)
    print("The loss is: {0:>10.5}\nThe accuracy is: {1:>10.5%}".format(loss, acc))
    

def predict(model_path):
    model = load_model(model_path)
    img_name_list = os.listdir(DEFAULT_TEST_PATH)
    result = []
    name = []
    for i, img_name in enumerate(img_name_list):
        im = Image.open(DEFAULT_TEST_PATH + "/" + img_name)
        print("predict " + img_name + ", {0}/{1}".format(i, len(img_name_list)))
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
    print("Command: ", args.command)
    print("Model: ", args.model)
    if args.command == "train":
        train(model_path=args.model, personal_model=args.pm)
    elif args.command == "evaluate":
        assert args.model is not None, "Please load a model..."
        evaluate(args.model)
    elif args.command == "predict":
        assert args.model is not None, "Please load a model..."
        predict(args.model)
    elif args.command == "debug":
        debug(args.model)
