import random
import sys

import numpy as np
from utils import *

lrate = 0.001
savedir = "model1/" + str(int(lrate * 1000))
np.set_printoptions(threshold=sys.maxsize)

inputs = np.load("dataset.npy")
targets = np.load("targets.npy")

# initialize random weights and biases
w1 = np.transpose([[random.uniform(-1, 1) for i in range(100)] for j in range(50)])
w2 = np.transpose([[random.uniform(-1, 1) for i in range(50)] for j in range(3)])
b1 = np.random.rand(1)
b1 = np.transpose(np.ones((50, 1)) * b1)[0]
b2 = np.random.rand(1)
b2 = np.transpose(np.ones((3, 1)) * b2)[0]

totalitr = 1000000
itr = 0

while (itr < totalitr):

    # forward propagation
    z1 = np.dot(inputs, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    cost = np.mean(np.square(a2 - targets))

    # calculating error
    delw2 = (-1 * lrate * np.transpose(a1)) @ (2 * (a2 - targets) * a2 * (1 - a2))
    delw1 = (-1 * lrate * np.transpose(inputs)) @ (2 * (a2 - targets) * a2 * (1 - a2)) @ np.transpose(w2)
    delb2 = -lrate * (2 * (a2 - targets) * a2 * (1 - a2))
    delb1 = -lrate * (2 * (a2 - targets) * a2 * (1 - a2)) @ np.transpose(w2)

    # updation
    w2 = w2 + delw2
    w1 = w1 + delw1
    b2 = b2 + np.mean(delb2, axis=0)
    b1 = b1 + np.mean(delb1, axis=0)

    itr += 1

    if (itr % 10000 == 0):
        print(itr * 100 / totalitr)
        print(cost)

np.save(savedir + "weights1", w1)
np.save(savedir + "weights2", w2)
np.save(savedir + "bias1", b1)
np.save(savedir + "bias2", b2)
