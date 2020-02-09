import gc

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

from tf_model import cnn_model
from tf_train_util import BengaliImageGenerator


def test_train_generator(img_generator: BengaliImageGenerator):
    n_imgs = 5
    x, y = img_generator[1]
    fig, axs = plt.subplots(n_imgs, 1, figsize=(10, 5 * n_imgs))
    print(x.shape)
    print(y["root"].shape)
    print(y["vowel"].shape)
    print(y["consonant"].shape)
    for i in range(n_imgs):
        img = x[i]

        axs[i].imshow(img.reshape(128, 128))
        axs[i].set_title('Original image')
        axs[i].axis('off')

    plt.show()


def train_tf():
    # need to set this to train in rtx gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    train = pd.read_csv("data/train.csv")
    train["image_path"] = train["image_id"].apply(lambda x: f"data/image/{x}.png")
    train.drop(["grapheme", "image_id"], axis=1, inplace=True)

    x = train["image_path"].values
    y_root = pd.get_dummies(train['grapheme_root']).values
    y_vowel = pd.get_dummies(train['vowel_diacritic']).values
    y_consonant = pd.get_dummies(train['consonant_diacritic']).values

    del train
    gc.collect()

    # img_generator = BengaliImageGenerator(x, root=y_root, vowel=y_vowel, consonant=y_consonant)
    # test_train_generator(img_generator)

    batch_size = 256

    kfold = KFold(n_splits=5)
    for train_idx, test_idx in kfold.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train_root, y_test_root = y_root[train_idx], y_root[test_idx]
        y_train_consonant, y_test_consonant = y_consonant[train_idx], y_consonant[test_idx]
        y_train_vowel, y_test_vowel = y_vowel[train_idx], y_vowel[test_idx]

        train_gen = BengaliImageGenerator(
            x_train,
            root=y_train_root,
            vowel=y_train_vowel,
            consonant=y_train_consonant,
            batch_size=batch_size
        )

        model = cnn_model(128)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, epochs=1, steps_per_epoch=x_train.shape[0])


if __name__ == "__main__":
    train_tf()
