import os
import sys

import cv2
import numpy as np
from utils import *

np.set_printoptions(threshold=sys.maxsize)
learning_rate = 0.000


def get_model(i):
    # load weights

    weights = []
    bias = []
    dir = os.listdir("model" + str(i))
    dir = sorted(dir)
    for filename in dir:
        # print(filename[1:-4])
        if ("weights" in filename):
            weights.append(np.load("model" + str(i) + "/" + filename))
        elif ("bias" in filename):
            bias.append(np.load("model" + str(i) + "/" + filename))
    return weights, bias


def get_testset(folderpath):
    dir = os.listdir(folderpath)
    imgpath = dir[int(np.random.randint(0, len(dir) - 1, 1, int))]
    img = cv2.imread(os.path.join(folderpath, imgpath), 0)
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw = cv2.resize(im_bw, (10, 10))
    arr = np.zeros((10, 10), dtype=int)
    for j in range(10):
        for i in range(10):
            if im_bw[i][j] > 100:
                arr[i][j] = 1
    arr = arr.flatten()
    return arr, imgpath


def run_model(mname, weights, bias, input):
    zin1 = sigmoid(np.dot(input, weights[0]) + bias[0])
    zin2 = sigmoid(np.dot(zin1, weights[1]) + bias[1])
    if (mname == 1):
        return zin2
    else:
        zin3 = sigmoid(np.dot(zin2, weights[2]) + bias[2])
        return zin3


weights, bias = get_model(1)
fvec1, t1name = get_testset("testset")
print(run_model(1, weights, bias, fvec1))
print(t1name)
