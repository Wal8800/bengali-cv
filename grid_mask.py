import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# https://github.com/akuxcw/GridMask/blob/master/imagenet_grid/utils/grid.py
def grid_mask(img: np.ndarray, d1, d2, rotate=1, ratio=0.5, mode=1) -> np.ndarray:
    h = img.shape[0]
    w = img.shape[1]

    # 1.5 * h, 1.5 * w works fine with the squared images
    # But with rectangular input, the mask might not be able to recover back to the input image shape
    # A square mask with edge length equal to the diagnoal of the input image
    # will be able to cover all the image spot after the rotation. This is also the minimum square.
    hh = math.ceil((math.sqrt(h * h + w * w)))
    d = np.random.randint(d1, d2)
    # maybe use ceil? but i guess no big difference
    l = math.ceil(d * ratio)

    mask = np.ones((hh, hh), np.float32)
    st_h = np.random.randint(d)
    st_w = np.random.randint(d)

    # setting whole row to 0
    for i in range(-1, hh // d + 1):
        s = d * i + st_h
        t = s + l

        # the min and max is use to bound the starting and ending index within the mask.
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[s:t, :] *= 0

    # setting whole column to 0
    for i in range(-1, hh // d + 1):
        s = d * i + st_w
        t = s + l
        s = max(min(s, hh), 0)
        t = max(min(t, hh), 0)
        mask[:, s:t] *= 0

    r = np.random.randint(rotate)
    mask = Image.fromarray(np.uint8(mask))
    mask = mask.rotate(r)
    mask = np.asarray(mask)
    mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

    # 1 for normal grid, 0 for reverse grid mask
    if mode == 1:
        mask = 1 - mask
    img = img * mask

    return img


def test_grid_mask():
    img = cv2.imread("data/image_128/Train_0.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(np.max(img), np.min(img))
    img = grid_mask(img, 24, 56, ratio=0.4)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    test_grid_mask()
