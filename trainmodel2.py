import random
import sys

import numpy as np
from utils import *

lrate = 0.001
savedir = "model2/" + str(int(lrate * 1000))
np.set_printoptions(threshold=sys.maxsize)

inputs = np.load("dataset.npy")
targets = np.load("targets.npy")

# initialize random weights and biases
w1 = np.transpose([[random.uniform(-1, 1) for i in range(100)] for j in range(50)])
w2 = np.transpose([[random.uniform(-1, 1) for i in range(50)] for j in range(16)])
w3 = np.transpose([[random.uniform(-1, 1) for i in range(16)] for j in range(3)])
b1 = np.random.rand(1)
b1 = np.transpose(np.ones((50, 1)) * b1)[0]
b2 = np.random.rand(1)
b2 = np.transpose(np.ones((16, 1)) * b2)[0]
b3 = np.random.rand(1)
b3 = np.transpose(np.ones((3, 1)) * b3)[0]

totalitr = 1000000
itr = 0

while (itr < totalitr):

    # forward propagation
    z1 = np.dot(inputs, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3)

    cost = np.mean(np.square(a3 - targets))

    # calculating error
    delw3 = (-1 * lrate * np.transpose(a2)) @ (2 * (a3 - targets) * a3 * (1 - a3))
    delw2 = (-1 * lrate * np.transpose(a1)) @ (
            ((2 * (a3 - targets) * a3 * (1 - a3)) @ np.transpose(w3)) * (a2 * (1 - a2)))
    delw1 = (-1 * lrate * np.transpose(inputs)) @ (
            ((2 * (a3 - targets) * a3 * (1 - a3)) @ np.transpose(w3)) * (a2 * (1 - a2)) @ np.transpose(w2) *
            (a1 * (1 - a1)))
    delb3 = -lrate * (2 * (a3 - targets) * a3 * (1 - a3))
    delb2 = -delb3 @ np.transpose(w3) * (a2 * (1 - a2))
    delb1 = delb2 @ np.transpose(w2) * (a1 * (1 - a1))

    # updation
    w3 = w3 + delw3
    w2 = w2 + delw2
    w1 = w1 + delw1
    b3 = b3 + np.mean(delb3, axis=0)
    b2 = b2 + np.mean(delb2, axis=0)
    b1 = b1 + np.mean(delb1, axis=0)

    itr += 1

    if (itr % 10000 == 0):
        print(itr * 100 / totalitr)
        print(cost)

np.save(savedir + "weights1", w1)
np.save(savedir + "weights2", w2)
np.save(savedir + "weights3", w3)
np.save(savedir + "bias1", b1)
np.save(savedir + "bias2", b2)
np.save(savedir + "bias3", b3)
