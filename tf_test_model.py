import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model, Model

from tf_train_util import BengaliImageGenerator


def test_model():
    # need to set this to train in rtx gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
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

    sample_index = np.random.choice(len(x), len(x) // 10)
    print(len(sample_index))

    x = x[sample_index]
    y_root = y_root[sample_index]
    y_vowel = y_vowel[sample_index]
    y_consonant = y_consonant[sample_index]

    train_gen = BengaliImageGenerator(
        x,
        64,
        root=y_root,
        vowel=y_vowel,
        consonant=y_consonant,
        batch_size=64
    )

    model: Model = load_model("model/tf_model_aug_mix.h5")
    pred = model.predict(train_gen.get_all_images())

    root_prediction = np.argmax(pred[0], axis=1)
    vowel_pred = np.argmax(pred[1], axis=1)
    con_pred = np.argmax(pred[2], axis=1)

    print(classification_report(np.argmax(y_vowel, axis=1), vowel_pred))
    print(confusion_matrix(np.argmax(y_vowel, axis=1), vowel_pred, normalize='true'))


if __name__ == "__main__":
    test_model()
