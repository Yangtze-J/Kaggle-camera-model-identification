
from Augment.Operations import Operation
import cv2
import numpy as np
import random
from PIL import Image
from io import BytesIO
import jpeg4py as jpeg
import skimage.exposure
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']


def random_manipulation(img, manipulation=None):

    if manipulation is None:
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


class Opera(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, manipulation):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        # Set your custom operation's member variables here as required:
        self.manipulation = manipulation

    # Your class must implement the perform_operation method:
    def perform_operation(self, image, manipulated):
        # Start of code to perform custom image operation.
        if self.manipulation is "random":
            manipulation = random.choice(MANIPULATIONS)
        else:
            manipulation = self.manipulation

        if manipulation.startswith('jpg'):
            quality = int(manipulation[3:])
            out = BytesIO()
            image.save(out, format='jpeg', quality=quality)
            image = Image.open(out)
            del out

        elif manipulation.startswith('gamma'):
            gamma = float(manipulation[5:])
            im_decoded = np.array(image).astype('uint8')
            im_decoded = skimage.exposure.adjust_gamma(im_decoded, gamma)
            image = Image.fromarray(im_decoded)
            del im_decoded

        elif manipulation.startswith('bicubic'):
            scale = float(manipulation[7:])
            im_decoded = np.array(image).astype('uint8')
            im_decoded = cv2.resize(im_decoded, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image = Image.fromarray(im_decoded)
            del im_decoded
        else:
            assert False

        manipulated = 1
        return image, manipulated
