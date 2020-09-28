import os

import cv2
import numpy as np


def generate(inputs):
    images = []
    targets = []
    for filename in os.listdir(inputs):
        img = cv2.imread(os.path.join(inputs, filename), 0)
        if img is not None:
            (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            im_bw = cv2.resize(im_bw, (10, 10))
            arr = np.zeros((10, 10), dtype=int)
            for j in range(10):
                for i in range(10):
                    if im_bw[i][j] > 100:
                        arr[i][j] = 1
            # arr=1-arr
            arr = arr.flatten()

            tag = []
            if ("o" in filename):
                tag = [0, 0, 1]
            if ("t" in filename):
                tag = [0, 1, 0]
            if ("a" in filename):
                tag = [1, 0, 0]

            images.append(arr)
            targets.append(tag)
    return images, targets


images, targets = generate("inputs")
np.save("inputs/dataset", images)
np.save("inputs/targets", targets)
