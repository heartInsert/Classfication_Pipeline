from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random


class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        # self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
        #                    'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']
        self.transforms = ['rotate','shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) // 2
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.5, self.max_Magnitude),
            "shearY": np.linspace(0, 0.5, self.max_Magnitude),
            "translateX": np.linspace(0, 0.3, self.max_Magnitude),
            "translateY": np.linspace(0, 0.3, self.max_Magnitude),
            "rotate": np.linspace(-45, 45, self.max_Magnitude),
            "color": np.linspace(-0.9, 0.9, self.max_Magnitude),
            "posterize": np.round(np.linspace(8, 2, self.max_Magnitude), 0).astype(np.int),
            "solarize": np.linspace(256, 240, self.max_Magnitude),
            "contrast": np.linspace(0.0, 0.5, self.max_Magnitude),
            "sharpness": np.linspace(0.0, 0.9, self.max_Magnitude),
            "brightness": np.linspace(0.0, 0.3, self.max_Magnitude),
            "autocontrast": [0] * self.max_Magnitude,
            "equalize": [0] * self.max_Magnitude,
            "invert": [0] * self.max_Magnitude
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: img,
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        # return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]
        return list(zip(sampled_ops, M))

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

    def test_single_operation(self, image, op_name, M=-1):
        '''
        :param image: image
        :param op_name: operation name in   self.transforms
        :param M: -1  stands  for the  max   Magnitude  in  there operation
        :return:
        '''
        operation = self.func[op_name]
        mag = self.ranges[op_name][M]
        image = operation(image, mag)
        return image


if __name__ == '__main__':
    # # this  is  for  call the whole fun
    # img_augment = Rand_Augment()
    # img_origal = Image.open(r'0a38b552372d.png')
    # img_final = img_augment(img_origal)
    # plt.imshow(img_final)
    # plt.show()
    # print('how to  call')
    max_Magnitude = 20
    # this  is for  a  single  fun  you  want to test
    img_augment = Rand_Augment(max_Magnitude=max_Magnitude)
    img_origal = Image.open(
        r'/home/xjz/Desktop/Coding/DL_Data/cassava_leaf_disease_classification/test_images/2216849948.jpg')
    for i in range(0, max_Magnitude):
        img_final = img_augment.test_single_operation(img_origal, 'solarize', M=i)
        plt.subplot(max_Magnitude // 2, 2, i + 1)
        plt.imshow(img_final)
    plt.show()
    print('how  to test')
