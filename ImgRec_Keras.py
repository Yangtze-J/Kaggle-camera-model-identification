from config import *
from model import fine_tune_inceptionresnet_v2, model_create


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


def train(model_path=None, personal_model=None):

    if model_path is None:
        if personal_model is True:
            model = model_create()
        else:
            model = fine_tune_inceptionresnet_v2()
    else:
        model = load_model(model_path)

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

    p.flip_top_bottom(probability=0.1)
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
    # which we define here in a variable named `train_batch_size`.
    # This is used later to define number of steps per epoch,
    # so it is best to keep it stored as a variable.

    pg = p.keras_generator(batch_size=train_batch_size)

    # The generator can now be used to created augmented data.
    # In Python, generators are invoked using the `next()` function -
    # the Augmentor generators will return images indefinitely,
    # and so `next()` can be called as often as required.
    #

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
    iteration = 0
    while True:
        iteration += 1
        print()
        print('-' * 50)
        print('Iteration', iteration)
        # steps_per_epoch=len(p.augmentor_images) / train_batch_size
        h = model.fit_generator(generator=pg, steps_per_epoch=len(p.augmentor_images)/train_batch_size,
                                epochs=1, verbose=1,
                                callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                                                         verbose=1, mode='auto')],
                                validation_data=vg, validation_steps=len(v.augmentor_images)/val_batch_size)
        print('Model learning rate :', K.get_value(model.optimizer.lr))
        acc = h.history['acc']
        loss = h.history['loss']
        if os.path.exists(DEFAULT_WEIGHT_PATH) is False:
            os.makedirs(DEFAULT_WEIGHT_PATH)
        model.save(DEFAULT_WEIGHT_PATH+"/new_model.h5")
        print("Iteration{0}: ,saved model".format(iteration))
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
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Personal Model:", args.pm)
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
