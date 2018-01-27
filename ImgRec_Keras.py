from config import *
from keras.applications import *
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Reshape, Flatten
from model import model_create
from manipulation import Opera


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
parser.add_argument('-p', '--pooling', type=str, default='avg', help='Type of pooling to use')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Number of GPUs to use')
parser.add_argument('-cs', '--crop-size', type=int, default=221, help='Crop size')
parser.add_argument('-me', '--max-epoch', type=int, default=500, help='Epoch to run')
parser.add_argument('-dpo', '--dropout', type=float, default=0.2, help='Dropout rate for FC layers')

args = parser.parse_args()

CROP_SIZE = args.crop_size
input_image_shape = (CROP_SIZE, CROP_SIZE, 3)
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']


def train(model_path=None, personal_model=None):

    if model_path is None:
        if personal_model is True:
            model = model_create()
            model_name = "personal_model"
        else:
            classifier = globals()[args.classifier]
            base_model = classifier(include_top=False,
                                    weights='imagenet',
                                    input_shape=input_image_shape)
                                    # pooling=args.pooling if args.pooling != 'none' else None)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            # x = Reshape((-1,))(x)
            # x = Flatten()(x)
            # let's add a fully-connected layer
            x = Dense(512, activation='relu', name='fc1')(x)
            x = Dropout(args.dropout, name='dropout_fc1')(x)
            x = Dense(128, activation='relu', name='fc2')(x)
            x = Dropout(args.dropout, name='dropout_fc2')(x)
            # x = Dense(2048, activation='relu')(x)
            # and a logistic layer -- let's say we have num_classes classes
            predictions = Dense(num_classes, activation='softmax')(x)
            # # this is the model we will train
            model = Model(inputs=base_model.input, outputs=predictions)
            # first: train only the top layers (which were randomly initialized)
            # i.e. freeze all convolutional Xception layers
            for layer in base_model.layers:
                layer.trainable = True
            model.summary()
            print(args.classifier + " Model Created")
            model_name = args.classifier
        last_epoch = 0
    else:
        model = load_model(model_path, compile=False)
        match = re.search(r'model/(.*)-epoch:(\d+)-(\d+.\d+)-(\d+.\d+).h5', args.model)
        model_name = match.group(1)
        last_epoch = int(match.group(2))
        print("Model name:{0}, last epoch:{1}".format(model_name, last_epoch))
    if args.gpus >= 2:
        model = multi_gpu_model(model, gpus=args.gpus)

    # opt = keras.optimizers.Adam(lr=0.001)
    opt = keras.optimizers.Nadam(lr=0.002)
    # opt = keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
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
    # steps_per_epoch = len(p.augmentor_images) / train_batch_size
    h = model.fit_generator(generator=pg, steps_per_epoch=50,
                            epochs=args.max_epoch, verbose=1,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                                                     verbose=1, mode='auto'),
                                       keras.callbacks.ModelCheckpoint(DEFAULT_WEIGHT_PATH+"/"+model_name+
                                                                       "-epoch:"+"{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.h5",
                                                                       monitor='val_loss', verbose=1,
                                                                       save_best_only=True, save_weights_only=False,
                                                                       mode='auto', period=1)],
                            validation_data=vg, validation_steps=len(v.augmentor_images)/val_batch_size,
                            initial_epoch=last_epoch)
    print('Model learning rate :', K.get_value(model.optimizer.lr))
    acc = h.history['acc']
    loss = h.history['loss']
    if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
        os.makedirs(DEFAULT_WEIGHT_PATH)
    # model.save(DEFAULT_WEIGHT_PATH+"/new_model.h5")
    log_results('bin_', acc, loss)


def all_trainable(model_path):
    model = load_model(model_path)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
        layer.trainable = True
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    model.save(model_path)


def debug2():
    p = Augmentor.Pipeline(DEFAULT_VAL_PATH)
    # clean not jpg image
    for augmentor_image in p.augmentor_images:
        with Image.open(augmentor_image.image_path) as opened_image:
            if opened_image.format is not 'JPEG':
                p.augmentor_images.remove(augmentor_image)

    width = input_image_shape[0]
    height = input_image_shape[1]

    manipu = Opera(probability=1, manipulation="random")
    # manipu = Opera(probability=1, manipulation=MANIPULATIONS[0])

    # p.flip_top_bottom(probability=0.1)
    p.add_operation(manipu)
    # because of bicubic operation, crop must be at least
    p.crop_by_size(probability=1, width=1024, height=1024, centre=False)

    p.status()

    pg = p.keras_generator(batch_size=train_batch_size)
    images, labels, origin = next(pg)
    for i in range(len(images)):
        img = Image.fromarray((images[i]*255).astype('uint8'), 'RGB')
        img.show()
        Ori = Image.fromarray((origin[i]*255).astype('uint8'), 'RGB')
        Ori.show()
    len(p.augmentor_images)


def debug1():
    # direct
    img_name_list = os.listdir(DEFAULT_VAL_PATH)

    for i, img_name in enumerate(img_name_list):
        imgs = os.listdir(DEFAULT_VAL_PATH + "/" + img_name)
        for imgg in imgs:
            im1 = Image.open(DEFAULT_VAL_PATH + "/" + img_name+"/"+imgg)
            im = np.asarray(im1).astype('float32')
            im = im.astype('uint8')
            im = Image.fromarray(im, 'RGB')
            im1.show()
            im.show()


def debug():

    keras.applications.densenet.DenseNet201(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                            pooling=None, classes=1000)


def add_one_dense(model_path):
    model = load_model(model_path)
    model.summary()

    fc1 = model.layers[-2]
    prediction = model.layers[-1]
    fc1.name = 'dense_1'
    prediction.name = 'prediction'
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu', name='dense_2')(fc1.output)
    # and a logistic layer -- let's say we have num_classes classes
    pred = prediction(x)

    # predictions = Dense(num_classes, activation='softmax')(x)
    # # this is the model we will train
    model = Model(inputs=model.input, outputs=pred)
    model.summary()
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
    import skimage.exposure
    with open(DEFAULT_LOG_PATH + '/' + filename + 'loss.csv', 'a', newline='') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow([loss_item])


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
