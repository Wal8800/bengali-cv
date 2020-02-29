import json
import sys
from csv import DictWriter

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence

import augmentations
from grid_mask import grid_mask
from random_eraser import get_random_eraser

np.set_printoptions(threshold=sys.maxsize)

# https://imgaug.readthedocs.io/en/latest/source/examples_basics.html#a-simple-and-common-augmentation-sequence
seq = iaa.Sequential([
    # iaa.Fliplr(0.5),  # horizontal flips
    # iaa.Crop(percent=(0, 0.1)),  # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    # iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)  # apply augmenters in random order

eraser = get_random_eraser(s_h=0.2, v_h=0, r_2=1 / 0.5)


def read_image(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    return img


class BengaliImageGenerator(Sequence):
    def __init__(self, file_paths, image_size, root, vowel, consonant, batch_size=64, shuffle=False):
        """Initialization
        :param file_paths: list of all 'label' ids to use in the generato
        """
        self.file_paths = file_paths
        self.root = root
        self.vowel = vowel
        self.consonant = consonant
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        x_batch = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        images = [read_image(p) for p in x_batch]
        result = np.array(images).reshape((-1, self.image_size, self.image_size, 1))

        root_batch = self.root[index * self.batch_size:(index + 1) * self.batch_size]
        vowel_batch = self.vowel[index * self.batch_size:(index + 1) * self.batch_size]
        consonant_batch = self.consonant[index * self.batch_size:(index + 1) * self.batch_size]

        y_dict = {
            'root': root_batch,
            'vowel': vowel_batch,
            'consonant': consonant_batch
        }

        return result, y_dict

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        p = np.random.permutation(len(self.file_paths))

        self.file_paths = self.file_paths[p]
        self.root = self.root[p]
        self.vowel = self.vowel[p]
        self.consonant = self.consonant[p]


class BengaliImageMixUpGenerator(BengaliImageGenerator):
    def __init__(self, file_paths, image_size, root=None, vowel=None, consonant=None, batch_size=64, shuffle=False,
                 alpha=0.2, mixup=True, transformers=None):
        """Initialization
        :param file_paths: list of all 'label' ids to use in the generato
        """
        super().__init__(file_paths, image_size, root, vowel, consonant, batch_size, shuffle)
        if transformers is None:
            transformers = []
        self.alpha = alpha
        self.mixup = mixup
        self.transformers = transformers

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        images, y_dict = super().__getitem__(index)

        if not self.mixup:
            for i in range(len(images)):
                img = images[i]
                img = img.reshape((self.image_size, self.image_size))

                for transformer in self.transformers:
                    img = transformer(img)

                img = img.reshape((self.image_size, self.image_size, 1))
                images[i] = img
            return images, y_dict

        current_batch_size = len(images)
        random_index = np.random.choice(len(self.file_paths), current_batch_size)
        x_batch_2 = self.file_paths[random_index]
        images_2 = [read_image(p) for p in x_batch_2]
        images_2 = np.array(images_2).reshape((-1, self.image_size, self.image_size, 1))

        lam = np.random.beta(self.alpha, self.alpha, current_batch_size)

        x_l = lam.reshape((current_batch_size, 1, 1, 1))
        y_l = lam.reshape((current_batch_size, 1))

        result_images = images * x_l + images_2 * (1 - x_l)
        for i, _ in enumerate(result_images):
            img = result_images[i]
            img = img.reshape((self.image_size, self.image_size))

            for transformer in self.transformers:
                img = transformer(img)

            img = img.reshape((self.image_size, self.image_size, 1))
            result_images[i] = img

        root_batch_2 = self.root[random_index]
        vowel_batch_2 = self.vowel[random_index]
        consonant_batch_2 = self.consonant[random_index]

        result_root_batch = y_dict['root'] * y_l + root_batch_2 * (1 - y_l)
        result_vowel_batch = y_dict['vowel'] * y_l + vowel_batch_2 * (1 - y_l)
        result_con_batch = y_dict['consonant'] * y_l + consonant_batch_2 * (1 - y_l)

        y_dict = {
            'root': result_root_batch,
            'vowel': result_vowel_batch,
            'consonant': result_con_batch
        }

        return result_images, y_dict

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        p = np.random.permutation(len(self.file_paths))

        self.file_paths = self.file_paths[p]
        self.root = self.root[p]
        self.vowel = self.vowel[p]
        self.consonant = self.consonant[p]


def test_train_generatorby_class(gen: Sequence, root_class: int):
    n_imgs = 6
    fig, axs = plt.subplots(n_imgs, 1, figsize=(10, 5 * n_imgs))
    i = 0
    for k in range(len(gen)):
        data_x, y = gen[k]

        for j in range(len(data_x)):
            if root_class not in np.nonzero(y['root'][j])[0]:
                continue

            img = data_x[j]
            axs[i].imshow(img.reshape(gen.image_size, gen.image_size))
            axs[i].set_title('Original image')
            axs[i].axis('off')
            i += 1
            if i >= n_imgs:
                break

        if i >= n_imgs:
            break

    plt.show()


def test_train_generator(gen: Sequence):
    n_imgs = 6
    print(np.random.choice(len(gen), 1)[0])
    data_x, y = gen[np.random.choice(len(gen), 1)[0]]
    fig, axs = plt.subplots(n_imgs, 1, figsize=(10, 5 * n_imgs))
    print(data_x.shape)
    print(y["root"].shape)
    print(y["vowel"].shape)
    print(y["consonant"].shape)
    i = 0
    for j in range(len(data_x)):
        if i >= n_imgs:
            break

        img = data_x[j]
        print(np.nonzero(y['root'][j]), np.nonzero(y['vowel'][j]), np.nonzero(y['consonant'][j]))
        if i == 0:
            # print(img)
            print(np.max(img), np.min(img))
        axs[i].imshow(img.reshape(gen.image_size, gen.image_size))
        axs[i].set_title('Original image')
        axs[i].axis('off')
        i += 1

    plt.show()


def test_generator():
    current_image_size = 128
    augmentations.IMAGE_SIZE = current_image_size
    train = pd.read_csv("data/train.csv")
    train["image_path"] = train["image_id"].apply(lambda x: f"data/image_{current_image_size}/{x}.png")
    train.drop(["grapheme", "image_id"], axis=1, inplace=True)

    image_x = train["image_path"].values
    y_root = pd.get_dummies(train['grapheme_root']).values
    y_vowel = pd.get_dummies(train['vowel_diacritic']).values
    y_consonant = pd.get_dummies(train['consonant_diacritic']).values

    img_generator = BengaliImageMixUpGenerator(
        image_x,
        current_image_size,
        root=y_root,
        vowel=y_vowel,
        consonant=y_consonant,
        transformers=[
            # lambda x: augmentations.augment_and_mix(x, depth=1),
            lambda x: grid_mask(x, d1=30, d2=75, ratio=0.5, rotate=360)
        ]
    )
    test_train_generator(img_generator)


def test_write_csv():
    info = {
        "image_size": 128,
        "batch_size": 128,
        "starting_lr": 0.01,
        "epoch": 1,
        "lr_reduce_patience": 5,
        "lr)reduce_factor": 0.7,
        "min_lr": 0.0000001,
        "augmentation": json.dumps({"hellO": 0}),
        "cv_score": 1.0,
        "public_cv": "",

    }

    with open("train_result.csv", 'a+') as write_obj:
        dict_writer = DictWriter(write_obj, fieldnames=list(info.keys()))

        dict_writer.writeheader()
        dict_writer.writerow(info)


if __name__ == "__main__":
    # test_generator()
    test_write_csv()
