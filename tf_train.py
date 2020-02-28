from collections import Counter
from datetime import datetime

import sklearn
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tf_model import *
from tf_train_util import *


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, generator: Sequence):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        # val_root_accuracy = logs["val_root_accuracy"]
        # if val_root_accuracy > 0.9:
        #     self.generator.aug_prob = 1.0
        # elif val_root_accuracy > 0.8:
        #     self.generator.aug_prob = 0.9
        # elif val_root_accuracy > 0.7:
        #     self.generator.aug_prob = 0.8
        # elif val_root_accuracy > 0.6:
        #     self.generator.aug_prob = 0.7
        # elif val_root_accuracy > 0.5:
        #     self.generator.aug_prob = 0.6

        self.generator.on_epoch_end()


def train_tf(image_size=64, batch_size=128, lr=0.001, min_lr=0.00001, epoch=30):
    augmentations.IMAGE_SIZE = image_size
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
    train["image_path"] = train["image_id"].apply(lambda x: f"data/image_{image_size}/{x}.png")
    train.drop(["grapheme", "image_id"], axis=1, inplace=True)

    x = train["image_path"].values
    y_root = pd.get_dummies(train['grapheme_root']).values
    y_vowel = pd.get_dummies(train['vowel_diacritic']).values
    y_consonant = pd.get_dummies(train['consonant_diacritic']).values
    print(
        f"overall dataset: "
        f"root - {len(np.unique(train['grapheme_root'].values))} "
        f"vowel - {len(np.unique(train['vowel_diacritic'].values))} "
        f"con - {len(np.unique(train['consonant_diacritic'].values))}"
    )
    # sample_index = np.random.choice(len(x), len(x) // 10)
    # print(len(sample_index))
    #
    # x = x[sample_index]
    # y_root = y_root[sample_index]
    # y_vowel = y_vowel[sample_index]
    # y_consonant = y_consonant[sample_index]

    # img_generator = BengaliImageGenerator(x, root=y_root, vowel=y_vowel, consonant=y_consonant)
    # test_train_generator(img_generator)

    kfold = KFold(n_splits=10)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in skf.split(x, train['grapheme_root'].values):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train_root, y_test_root = y_root[train_idx], y_root[test_idx]
        y_train_consonant, y_test_consonant = y_consonant[train_idx], y_consonant[test_idx]
        y_train_vowel, y_test_vowel = y_vowel[train_idx], y_vowel[test_idx]

        root_truth = np.argmax(y_test_root, axis=1)
        vowel_truth = np.argmax(y_test_vowel, axis=1)
        con_truth = np.argmax(y_test_consonant, axis=1)

        print(
            f"train set: "
            f"root - {len(np.unique(np.argmax(y_train_root, axis=1)))} "
            f"vowel - {len(np.unique(np.argmax(y_train_vowel, axis=1)))} "
            f"con - {len(np.unique(np.argmax(y_train_consonant, axis=1)))}"
        )

        print(
            f"test set: "
            f"root - {len(np.unique(root_truth))} "
            f"vowel - {len(np.unique(vowel_truth))} "
            f"con - {len(np.unique(con_truth))}"
        )
        c = Counter(vowel_truth)
        vowel_test_percentage = [(i, c[i] / len(vowel_truth) * 100.0) for i, count in c.most_common()]
        print(f"test vowel percentage: {vowel_test_percentage}")
        c = Counter(con_truth)
        con_test_percentage = [(i, c[i] / len(con_truth) * 100.0) for i, count in c.most_common()]
        print(f"test con percentage: {con_test_percentage}")

        train_gen = BengaliImageMixUpGenerator(
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
            batch_size=batch_size,
            augment=False
        )

        logdir = f"logs/scalars/{image_size}/mixup_gridmask/" + datetime.now().strftime("%Y%m%d-%H%M%S")

        model = dense_net_121_model(image_size)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_consonant_accuracy', patience=3, verbose=1, factor=0.7, min_lr=min_lr),
            ReduceLROnPlateau(monitor='val_root_accuracy', patience=3, verbose=1, factor=0.7, min_lr=min_lr),
            ReduceLROnPlateau(monitor='val_vowel_accuracy', patience=3, verbose=1, factor=0.7, min_lr=min_lr),
            OnEpochEnd(train_gen),
            TensorBoard(log_dir=logdir),
            ModelCheckpoint(f"model/tf_model_imgaug_{image_size}_mixup_gridmask.h5", monitor='val_root_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
        ]

        # {'loss': 3.230544132721944, 'root_loss': 2.2817774, 'vowel_loss': 0.5255275, 'consonant_loss': 0.42362353,
        #  'root_accuracy': 0.38500798, 'vowel_accuracy': 0.82589376, 'consonant_accuracy': 0.8602432,
        #  'val_loss': 1.568948200933493, 'val_root_loss': 1.1511563, 'val_vowel_loss': 0.21462844,
        #  'val_consonant_loss': 0.20316331, 'val_root_accuracy': 0.6505925, 'val_vowel_accuracy': 0.9300189,
        #  'val_consonant_accuracy': 0.9365664}

        model.fit(train_gen, epochs=epoch, callbacks=callbacks, validation_data=test_gen)

        prediction = model.predict(test_gen)
        scores = []
        for x in prediction:
            print(x.shape)

        root_prediction = np.argmax(prediction[0], axis=1)
        scores.append(sklearn.metrics.recall_score(root_prediction, root_truth, average='macro'))
        print(classification_report(root_truth, root_prediction))
        vowel_pred = np.argmax(prediction[1], axis=1)
        scores.append(sklearn.metrics.recall_score(vowel_pred, vowel_truth, average='macro'))
        print(classification_report(vowel_truth, vowel_pred))
        con_pred = np.argmax(prediction[2], axis=1)
        scores.append(sklearn.metrics.recall_score(con_pred, con_truth, average='macro'))
        print(classification_report(con_truth, con_pred))
        print(np.average(scores, weights=[2, 1, 1]))
        break


if __name__ == "__main__":
    train_tf(image_size=128, batch_size=128, lr=0.001, epoch=50, min_lr=0.000001)
