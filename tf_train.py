from collections import Counter
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import classification_report, recall_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tf_model import *
from tf_train_util import *

GRID_MASK = "gridmask"
MIXUP = "mixup"


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, generator: Sequence):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end()


def train_tf(image_size=64, batch_size=128, lr=0.001, min_lr=0.00001, epoch=30, logging=True, save_model=True,
             save_result_to_csv=True, lr_reduce_patience=5, lr_reduce_factor=0.7, n_fold=5, aug_config=None):
    if aug_config is None:
        aug_config = dict()
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

    transformers = []
    if GRID_MASK in aug_config:
        transformers.append(lambda img: grid_mask(img, **aug_config[GRID_MASK]))

    skf = StratifiedKFold(n_splits=n_fold, shuffle=True)
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
            batch_size=batch_size,
            mixup=MIXUP in aug_config,
            alpha=aug_config[MIXUP]["alpha"] if MIXUP in aug_config else 0.2,
            transformers=transformers
        )

        test_gen = BengaliImageGenerator(
            x_test,
            image_size,
            root=y_test_root,
            vowel=y_test_vowel,
            consonant=y_test_consonant,
            batch_size=batch_size
        )

        model = dense_net_121_model(image_size)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_consonant_accuracy', patience=lr_reduce_patience, verbose=1,
                              factor=lr_reduce_factor, min_lr=min_lr),
            ReduceLROnPlateau(monitor='val_root_accuracy', patience=lr_reduce_patience, verbose=1,
                              factor=lr_reduce_factor, min_lr=min_lr),
            ReduceLROnPlateau(monitor='val_vowel_accuracy', patience=lr_reduce_patience, verbose=1,
                              factor=lr_reduce_factor, min_lr=min_lr),
            OnEpochEnd(train_gen),
        ]

        aug_keys = list(aug_config.keys())
        aug_keys.sort(reverse=True)
        if len(aug_keys) == 0:
            aug_keys.append("base")

        if logging:
            logdir = f"logs/scalars/{image_size}/{'_'.join(aug_keys)}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(TensorBoard(log_dir=logdir))

        if save_model:
            callbacks.append(ModelCheckpoint(f"model/tf_model_imgaug_{image_size}_{'_'.join(aug_keys)}.h5",
                                             monitor='val_root_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max'))

        model.fit(train_gen, epochs=epoch, callbacks=callbacks, validation_data=test_gen)

        prediction = model.predict(test_gen)
        scores = []

        root_prediction = np.argmax(prediction[0], axis=1)
        scores.append(recall_score(root_prediction, root_truth, average='macro'))
        print(classification_report(root_truth, root_prediction))
        vowel_pred = np.argmax(prediction[1], axis=1)
        scores.append(recall_score(vowel_pred, vowel_truth, average='macro'))
        print(classification_report(vowel_truth, vowel_pred))
        con_pred = np.argmax(prediction[2], axis=1)
        scores.append(recall_score(con_pred, con_truth, average='macro'))
        print(classification_report(con_truth, con_pred))

        cv_score = np.average(scores, weights=[2, 1, 1])
        print(cv_score)

        if save_result_to_csv:
            info = {
                "image_size": image_size,
                "batch_size": batch_size,
                "starting_lr": lr,
                "epoch": epoch,
                "lr_reduce_patience": lr_reduce_patience,
                "lr)reduce_factor": lr_reduce_factor,
                "min_lr": min_lr,
                "augmentation": json.dumps(aug_config),
                "cv_score": cv_score,
                "public_cv": "",

            }

            with open("train_result.csv", 'a+') as write_obj:
                dict_writer = DictWriter(write_obj, fieldnames=list(info.keys()))
                dict_writer.writerow(info)

        break


if __name__ == "__main__":
    img_aug_config = {
        GRID_MASK: {"d1": 30, "d2": 75, "ratio": 0.5, "rotate": 360},
        MIXUP: {"alpha": 0.2}
    }

    train_tf(
        image_size=64,
        batch_size=256,
        lr=0.001,
        epoch=30,
        min_lr=0.000001,
        save_model=True,
        logging=True,
        save_result_to_csv=True,
        aug_config=img_aug_config
    )
