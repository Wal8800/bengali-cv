import cv2
import numpy as np
import pandas as pd
import sklearn
from tqdm import tqdm

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


def process_and_output():
    trains = [
        'data/train_image_data_0.parquet',
        'data/train_image_data_1.parquet',
        'data/train_image_data_2.parquet',
        'data/train_image_data_3.parquet',
    ]

    for file_path in trains:
        df = pd.read_parquet(file_path)
        data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(np.uint8)
        for idx in tqdm(range(len(df))):
            name = df.iloc[idx, 0]
            # normalize each image by its max val
            img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
            img = crop_resize(img)

            # img = cv2.imencode('.png', img)[1]
            cv2.imwrite(f"data/image_128/{name}.png", img)


# https://www.kaggle.com/c/bengaliai-cv19/overview/evaluation
def score_func(solution, submission):
    scores = []
    for component in ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']:
        y_true_subset = solution[solution[component] == component]['target'].values
        y_pred_subset = submission[submission[component] == component]['target'].values
        scores.append(sklearn.metrics.recall_score(
            y_true_subset, y_pred_subset, average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])

    return final_score


def info():
    train = pd.read_csv("data/train.csv")
    print(train.head())
    print(train.shape)
    print("grapheme_root", train["grapheme_root"].nunique())
    print("grapheme_vowel", train["vowel_diacritic"].nunique())
    print("grapheme_dia", train["consonant_diacritic"].nunique())


if __name__ == "__main__":
    process_and_output()
