import sys

import cv2
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence

import augmentations
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


def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.


# https://github.com/google-research/augmix/blob/master/augment_and_mix.py
def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
      image: Raw input image as float32 np.ndarray of shape (h, w, c)
      severity: Severity of underlying augmentation operators (between 1 to 10).
      width: Width of augmentation chain
      depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
      alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
      mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        temp = ws[i] * image_aug
        mix += temp

    mixed = (1 - m) * image + m * mix
    return mixed


class BengaliImageGenerator(Sequence):
    def __init__(self, file_paths, image_size, root=None, vowel=None, consonant=None, batch_size=64, shuffle=False):
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
        images = []
        for p in x_batch:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255

            img = augment_and_mix(img)
            img = eraser(img.reshape((self.image_size, self.image_size, 1)))
            images.append(img)

            result = np.array(images).reshape((-1, self.image_size, self.image_size, 1))
            # result = seq(images=result)
            if self.root is None:
                return result

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
        self.file_paths = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.file_paths)

    def get_all_images(self):
        images = []
        for p in self.file_paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255

            images.append(img)

        return np.array(images).reshape((-1, self.image_size, self.image_size, 1))


def test_train_generator(gen: BengaliImageGenerator):
    n_imgs = 6
    print(np.random.choice(len(gen), 1)[0])
    data_x, y = gen[np.random.choice(len(gen), 1)[0]]
    fig, axs = plt.subplots(n_imgs, 1, figsize=(10, 5 * n_imgs))
    print(data_x.shape)
    print(y["root"].shape)
    print(y["vowel"].shape)
    print(y["consonant"].shape)
    for i in range(n_imgs):
        img = data_x[i]
        if i == 0:
            # print(img)
            print(np.max(img), np.min(img))
        axs[i].imshow(img.reshape(gen.image_size, gen.image_size))
        axs[i].set_title('Original image')
        axs[i].axis('off')

    plt.show()


if __name__ == "__main__":
    train = pd.read_csv("data/train.csv")
    train["image_path"] = train["image_id"].apply(lambda x: f"data/image_128/{x}.png")
    train.drop(["grapheme", "image_id"], axis=1, inplace=True)

    x = train["image_path"].values
    y_root = pd.get_dummies(train['grapheme_root']).values
    y_vowel = pd.get_dummies(train['vowel_diacritic']).values
    y_consonant = pd.get_dummies(train['consonant_diacritic']).values

    img_generator = BengaliImageGenerator(x, 128, root=y_root, vowel=y_vowel, consonant=y_consonant)
    test_train_generator(img_generator)
