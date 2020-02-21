import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import Sequence

HEIGHT = 137
WIDTH = 236
SIZE = 128


# Taken: https://www.kaggle.com/iafoss/image-preprocessing-128x128
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# Taken: https://www.kaggle.com/iafoss/image-preprocessing-128x128
def crop_resize(img0, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    l = max(lx, ly) + pad
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l - ly) // 2,), ((l - lx) // 2,)], mode='constant')
    return cv2.resize(img, (size, size))


is_kaggle = False


def test_predict_image(test_image_list):
    n_imgs = len(test_image_list)
    fig, axs = plt.subplots(n_imgs, 1, figsize=(10, 5 * n_imgs))
    for i in range(n_imgs):
        img = test_image_list[i]
        if i == 0:
            print(img)
            print(np.max(img), np.min(img))
        axs[i].imshow(img.reshape(64, 64))
        axs[i].set_title('Original image')
        axs[i].axis('off')

    plt.show()


class TestImageGenerator(Sequence):
    def __init__(self, file_paths, image_size, batch_size=128):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.file_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        x_batch = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        for p in x_batch:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img / 255
            images.append(img)

        return np.array(images).reshape((-1, self.image_size, self.image_size, 1))


def predict():
    if not is_kaggle:
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

    model_dir = "../input/bengalicvmodel" if is_kaggle else "../model"
    model: Model = load_model(f"{model_dir}/tf_model_imgaug_128.h5")

    tests = [
        "../input/bengaliai-cv19/test_image_data_0.parquet",
        "../input/bengaliai-cv19/test_image_data_1.parquet",
        "../input/bengaliai-cv19/test_image_data_2.parquet",
        "../input/bengaliai-cv19/test_image_data_3.parquet",
    ]

    target = []
    row_id = []
    for file_path in tests:
        print(f"processing: {file_path}")
        preds_dict = {
            'grapheme_root': [],
            'vowel_diacritic': [],
            'consonant_diacritic': []
        }

        df = pd.read_parquet(file_path)
        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)

        idx_batch = np.array_split(np.arange(len(df)), 5)
        for batch in idx_batch:
            test_image_list = []
            if len(batch) == 0:
                continue

            for idx in batch:
                # normalize each image by its max val
                img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
                img = crop_resize(img)
                img = img / 255

                # img = cv2.imencode('.png', img)[1]
                test_image_list.append(img)

            test_images = np.array(test_image_list).reshape((-1, SIZE, SIZE, 1))
            preds = model.predict(test_images)

            for i, p in enumerate(preds_dict):
                preds_dict[p] = np.argmax(preds[i], axis=1)

            for i, idx in enumerate(batch):
                name = df.iloc[idx, 0]
                for comp_name, comp_preds in preds_dict.items():
                    row_id.append(f"{name}_{comp_name}")
                    target.append(comp_preds[i])
            gc.collect()

    df_sample = pd.DataFrame(
        {
            'row_id': row_id,
            'target': target
        },
        columns=['row_id', 'target']
    )
    df_sample.to_csv('submission.csv', index=False)
    df_sample.head()


if __name__ == "__main__":
    predict()
