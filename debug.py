

def len_predict():
    img_name_list = os.listdir(DEFAULT_TEST_PATH)
    manip = 0
    unalt = 0
    for i, img_name in enumerate(img_name_list):
        print("predict " + img_name + ", {0}/{1}".format(i, len(img_name_list)))
        if img_name.find('manip') != -1:
            manip += 1
        else:
            unalt += 1

    print("Number of unalt:{0}, percent:{1}.".format(unalt, unalt/len(img_name_list)))
    print("Number of manip:{0}, percent:{1}.".format(manip, manip/len(img_name_list)))


def predict1(model_path):
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


def debug2():
    p = Pipeline(DEFAULT_VAL_PATH)
    # clean not jpg image
    for augmentor_image in p.augmentor_images:
        with Image.open(augmentor_image.image_path) as opened_image:
            if opened_image.format is not 'JPEG':
                p.augmentor_images.remove(augmentor_image)

    width = input_image_shape[0]
    height = input_image_shape[1]

    manipu = Opera(probability=0.5, manipulation="random")
    # manipu = Opera(probability=1, manipulation=MANIPULATIONS[0])

    # p.flip_top_bottom(probability=0.1)
    p.add_operation(manipu)
    # because of bicubic operation, crop must be at least
    p.crop_by_size(probability=1, width=128, height=128, centre=False)

    p.status()

    pg = p.keras_generator(batch_size=train_batch_size)

    [images, manipulated], labels = next(pg)

    print(manipulated)
    # for i in range(len(images)):
    #     img = Image.fromarray((images[i]*255).astype('uint8'), 'RGB')
    #     img.show()
    #     Ori = Image.fromarray((origin[i]*255).astype('uint8'), 'RGB')
    #     Ori.show()
    # len(p.augmentor_images)


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


def change_trainable(model_path):
    model = load_model(model_path)
    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
        layer.trainable = False

    model.summary()
    print(model.layers[-4].name)
    x = Flatten()(model.layers[-4].output)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(args.dropout, name='dropout_fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(args.dropout, name='dropout_fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dropout(args.dropout, name='dropout_fc3')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
    model.summary()
    model.save(DEFAULT_WEIGHT_PATH+'/Changed_Xception.h5')


def add_manipulation(model_path):
    model = load_model(model_path)
    # input_image = Input(shape=(CROP_SIZE, CROP_SIZE, 3))
    manipulated = Input(shape=(1,), name="manipulation")

    for i, layer in enumerate(model.layers):
        print(i, layer.name, layer.trainable)
        layer.trainable = False

    x = model.layers[-9]
    print(x.name)
    print(model.layers[-8].name)
    x = Reshape((-1,))(x.output)

    x = concatenate([x, manipulated])
    x = Dense(1024, activation='relu', name='fc1')(x)
    drop1 = model.layers[-6](x)
    fc2 = model.layers[-5](drop1)
    drop2 = model.layers[-4](fc2)
    fc3 = model.layers[-3](drop2)
    drop3 = model.layers[-2](fc3)
    prediction = model.layers[-1](drop3)

    model = Model(inputs=(model.input, manipulated), outputs=prediction)
    plot_model(model, to_file='manipulated_model.png')
    model.summary()
    model.save(DEFAULT_WEIGHT_PATH+'/XceptionManipu.h5')