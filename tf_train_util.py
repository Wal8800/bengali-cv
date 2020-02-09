import cv2
import imgaug.augmenters as iaa
import numpy as np
from tensorflow.keras.utils import Sequence

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])


class BengaliImageGenerator(Sequence):
    def __init__(self, file_paths, root=None, vowel=None, consonant=None, batch_size=64, shuffle=False):
        """Initialization
        :param file_paths: list of all 'label' ids to use in the generato
        """
        self.file_paths = file_paths
        self.root = root
        self.vowel = vowel
        self.consonant = consonant
        self.shuffle = shuffle
        self.batch_size = batch_size

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

            images.append(img)

        result = np.array(images).reshape((-1, 128, 128, 1))

        if self.root is None:
            return seq(images=result)

        root_batch = self.root[index * self.batch_size:(index + 1) * self.batch_size]
        vowel_batch = self.vowel[index * self.batch_size:(index + 1) * self.batch_size]
        consonant_batch = self.consonant[index * self.batch_size:(index + 1) * self.batch_size]

        y_dict = {
            'root': root_batch,
            'vowel': vowel_batch,
            'consonant': consonant_batch
        }

        return seq(images=result), y_dict

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.file_paths = np.arange(len(self.file_paths))
        if self.shuff:
            np.random.shuffle(self.file_paths)
