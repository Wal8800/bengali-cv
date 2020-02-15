import gc
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau

from tf_model import *
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

    # sample_index = np.random.choice(len(x), len(x) // 10)
    # print(len(sample_index))
    #
    # x = x[sample_index]
    # y_root = y_root[sample_index]
    # y_vowel = y_vowel[sample_index]
    # y_consonant = y_consonant[sample_index]

    del train
    gc.collect()

    # img_generator = BengaliImageGenerator(x, root=y_root, vowel=y_vowel, consonant=y_consonant)
    # test_train_generator(img_generator)

    batch_size = 128
    image_size = 64
    kfold = KFold(n_splits=10)
    for train_idx, test_idx in kfold.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train_root, y_test_root = y_root[train_idx], y_root[test_idx]
        y_train_consonant, y_test_consonant = y_consonant[train_idx], y_consonant[test_idx]
        y_train_vowel, y_test_vowel = y_vowel[train_idx], y_vowel[test_idx]

        train_gen = BengaliImageGenerator(
            x_train,
            image_size,
            root=y_train_root,
            vowel=y_train_vowel,
            consonant=y_train_consonant,
            batch_size=batch_size
        )

        test_gen = BengaliImageGenerator(
            x_test,
            image_size,
            root=y_test_root,
            vowel=y_test_vowel,
            consonant=y_test_consonant,
            batch_size=batch_size
        )

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        model = dense_net_101_model(image_size)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='consonant_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001),
            ReduceLROnPlateau(monitor='root_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001),
            ReduceLROnPlateau(monitor='vowel_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001),
            TensorBoard(log_dir=logdir),
            ModelCheckpoint("model/tf_model.h5", monitor='val_root_accuracy', verbose=1, save_best_only=True,
                            mode='max')
        ]

        all_test_images = test_gen.get_all_images()
        validation_data = (all_test_images, [y_test_root, y_test_vowel, y_test_consonant])
        model.fit(train_gen, epochs=30, callbacks=callbacks, validation_data=validation_data)

        prediction = model.predict(test_gen)
        scores = []
        for x in prediction:
            print(x.shape)

        root_prediction = np.argmax(prediction[0], axis=1)
        root_truth = np.argmax(y_test_root, axis=1)
        scores.append(sklearn.metrics.recall_score(root_prediction, root_truth, average='macro'))

        vowel_pred = np.argmax(prediction[1], axis=1)
        vowel_truth = np.argmax(y_test_vowel, axis=1)
        scores.append(sklearn.metrics.recall_score(vowel_pred, vowel_truth, average='macro'))

        con_pred = np.argmax(prediction[2], axis=1)
        con_truth = np.argmax(y_test_consonant, axis=1)
        scores.append(sklearn.metrics.recall_score(con_pred, con_truth, average='macro'))

        print(np.average(scores, weights=[2, 1, 1]))
        break


if __name__ == "__main__":
    train_tf()
