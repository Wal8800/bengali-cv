import json
from collections import Counter
from csv import DictWriter
from datetime import datetime
from pathlib import Path

from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from random_eraser import get_random_eraser
from tf_model import *
from tf_train_util import *

GRID_MASK = "gridmask"
MIXUP = "mixup"
RANDOM_CUTOUT = "randomcutout"
AUGMIX = "augmix"
AFFINE = "affine"


class OnEpochEnd(tf.keras.callbacks.Callback):
    def __init__(self, generator: Sequence):
        super().__init__()
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        self.generator.on_epoch_end()


def train_tf(image_size=64, batch_size=128, lr=0.001, min_lr=0.00001, epoch=30, logging=True, save_model=True,
             save_result_to_csv=True, lr_reduce_patience=5, lr_reduce_factor=0.7, n_fold=5, aug_config=None,
             create_model=dense_net_121_model, three_channel=False):
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
    else:
        raise NotImplementedError("can't train with gpu")

    if aug_config is None:
        aug_config = dict()
    augmentations.IMAGE_SIZE = image_size
    # need to set this to train in rtx gpu

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

    if AFFINE in aug_config:
        seq_augmenter = iaa.Sequential([
            iaa.Sometimes(1.0, iaa.Affine(**aug_config[AFFINE])),
        ])
        transformers.append(lambda img: seq_augmenter.augment_image(img))
    if GRID_MASK in aug_config:
        transformers.append(lambda img: grid_mask(img, **aug_config[GRID_MASK]))

    if RANDOM_CUTOUT in aug_config:
        transformers.append(lambda img: get_random_eraser(**aug_config[RANDOM_CUTOUT])(img))

    if AUGMIX in aug_config:
        transformers.append(lambda img: augmentations.augment_and_mix(img, **aug_config[AUGMIX]))

    """
        32, 0.79
        60, 0.58
        61, 0.68
        62, 0.73
        84, 0.80
        37, 0.86
        45, 0.86
        110, 0.87
        122, 0.85
    """

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

        input_shape = (image_size, image_size, 3) if three_channel else (image_size, image_size, 3)

        train_gen = BengaliImageMixUpGenerator(
            x_train,
            image_size,
            root=y_train_root,
            vowel=y_train_vowel,
            consonant=y_train_consonant,
            batch_size=batch_size,
            mixup=MIXUP in aug_config,
            alpha=aug_config[MIXUP]["alpha"] if MIXUP in aug_config else 0.2,
            transformers=transformers,
            three_channel=three_channel
        )

        test_gen = BengaliImageGenerator(
            x_test,
            image_size,
            root=y_test_root,
            vowel=y_test_vowel,
            consonant=y_test_consonant,
            batch_size=batch_size,
            three_channel=three_channel
        )

        model = create_model(input_shape)
        optimizer = Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks = [
            ReduceLROnPlateau(monitor='val_root_accuracy', patience=lr_reduce_patience, verbose=1,
                              factor=lr_reduce_factor, min_lr=min_lr),
            OnEpochEnd(train_gen),
        ]

        aug_keys = list(aug_config.keys())
        aug_keys.sort(reverse=True)
        if len(aug_keys) == 0:
            aug_keys.append("base")

        if logging:
            current_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logdir = f"logs/scalars/{image_size}/{create_model.__name__}/{'_'.join(aug_keys)}/{current_timestamp}"
            callbacks.append(TensorBoard(log_dir=logdir))

        if save_model:
            Path(f"model/{create_model.__name__}").mkdir(parents=True, exist_ok=True)
            callbacks.append(
                ModelCheckpoint(f"model/{create_model.__name__}/tf_model_{image_size}_{'_'.join(aug_keys)}.h5",
                                monitor='val_root_accuracy',
                                verbose=1,
                                save_best_only=True,
                                mode='max'))

        model.fit(train_gen, epochs=epoch, callbacks=callbacks, validation_data=test_gen)

        prediction = model.predict(test_gen)
        scores = []

        root_prediction = np.argmax(prediction[0], axis=1)
        scores.append(recall_score(root_truth, root_prediction, average='macro'))
        # print(classification_report(root_truth, root_prediction))
        vowel_pred = np.argmax(prediction[1], axis=1)
        scores.append(recall_score(vowel_truth, vowel_pred, average='macro'))
        # print(classification_report(vowel_truth, vowel_pred))
        con_pred = np.argmax(prediction[2], axis=1)
        scores.append(recall_score(con_truth, con_pred, average='macro'))
        # print(classification_report(con_truth, con_pred))

        cv_score = np.average(scores, weights=[2, 1, 1])
        print(cv_score)

        if save_result_to_csv:
            info = {
                "model": create_model.__name__,
                "image_size": image_size,
                "batch_size": batch_size,
                "starting_lr": lr,
                "epoch": epoch,
                "lr_reduce_patience": lr_reduce_patience,
                "lr_reduce_factor": lr_reduce_factor,
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
        AFFINE: {
            "scale": {"x": (0.9, 1.1), "y": (0.9, 1.1)},
            "translate_percent": {"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            "rotate": (-25, 25),
            "shear": (-8, 8)
        },
        GRID_MASK: {"d1": 30, "d2": 75, "ratio": 0.7, "rotate": 360},
        MIXUP: {"alpha": 0.2},
        # RANDOM_CUTOUT: {"p": 1.0, "s_h": 0.3, "v_h": 0, "r_1": 0.4, "r_2": 1 / 0.4},
        # AUGMIX: {"augmentation_list": augmentations.augmentations_subset}
    }

    train_tf(
        image_size=224,
        batch_size=40,
        lr=0.0001,
        epoch=80,
        min_lr=0.00001,
        save_model=True,
        logging=True,
        save_result_to_csv=True,
        aug_config=img_aug_config,
        lr_reduce_patience=5,
        lr_reduce_factor=0.75,
        create_model=dense_net_121_model,
        three_channel=True
    )
