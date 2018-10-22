"""
This module contains the gradient-based optimizers
"""

import numpy as np
lr = 0.01


class Optimizer:
    def __init__(self, name, mode, f, start_pos):
        self.name = name
        # function to minimize
        self.f = f
        # starting position for finding the minimum
        self.state = start_pos
        # optimizing mode, one of the below functions
        self.mode = mode
        # v and s start at (0,0)
        self.v = np.array([0, 0])
        self.s = np.array([0, 0])
        self.steps = 0

    def step(self):
        self.steps += 1
        self.state = self.mode(self)


def SGD(optimizer):
    #TODO
    new_state = optimizer.state
    return new_state


def AdaGrad(optimizer):
    #TODO
    new_state = optimizer.state
    return new_state


def NesterovMomentum(optimizer, miu=0.9):
    v = optimizer.v
    f = optimizer.f
    state = optimizer.state

    optimizer.v = miu * v + lr * f(state, derivative=True)
    new_state = state - optimizer.v
    return new_state


def RMSProp(optimizer, beta=0.1, eps=10**(-8)):
    state = optimizer.state
    g = optimizer.f(state, derivative=True)
    optimizer.s = (1 - beta) * optimizer.s + beta * g*g
    new_state = state - lr * g / (np.sqrt(np.asarray(optimizer.s)) + eps)
    return new_state


def Adam(optimizer, eps=10**(-8), beta1=0.9, beta2=0.999):
    v = optimizer.v
    s = optimizer.s
    state = optimizer.state
    g = optimizer.f(state, derivative=True)

    optimizer.v = beta1 * v + (1. - beta1) * g
    optimizer.s = beta2 * s + (1. - beta2) * g*g

    v_unbiased = v / (1. - beta1 ** optimizer.steps)
    s_unbiased = s / (1. - beta2 ** optimizer.steps)

    new_state = state - lr * v_unbiased / (np.sqrt(s_unbiased) + eps)
    return new_state

