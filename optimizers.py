"""
This module contains the gradient-based optimizers
"""

import numpy as np


class Optimizer:
    def __init__(self, name, mode, f, start_pos, lr=0.01, miu=0.9, beta=0.1,
                 beta1=0.9, beta2=0.999, eps=10**(-8)):
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

        self.lr = lr

        # for nesterov
        self.miu = miu

        # for rmsprop and adagrad
        self.beta = beta

        # for adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self):
        self.steps += 1
        self.state = self.mode(self)


def SGD(optimizer):
    # TODO
    new_state = optimizer.state
    return new_state


def AdaGrad(optimizer):
    # TODO
    new_state = optimizer.state
    return new_state


def NesterovMomentum(opti):
    v = opti.v
    f = opti.f
    state = opti.state

    opti.v = opti.miu * v + opti.lr * f(state, derivative=True)
    new_state = state - opti.v
    return new_state


def RMSProp(opti):
    state = opti.state
    g = opti.f(state, derivative=True)
    opti.s = (1 - opti.beta) * opti.s + opti.beta * g*g
    new_state = state - opti.lr * g / (np.sqrt(opti.s) + opti.eps)
    return new_state


def Adam(opti):
    v = opti.v
    s = opti.s
    state = opti.state
    g = opti.f(state, derivative=True)

    opti.v = opti.beta1 * v + (1. - opti.beta1) * g
    opti.s = opti.beta2 * s + (1. - opti.beta2) * g*g

    v_unbiased = v / (1. - opti.beta1 ** opti.steps)
    s_unbiased = s / (1. - opti.beta2 ** opti.steps)

    new_state = state - opti.lr * v_unbiased / (np.sqrt(s_unbiased) + opti.eps)
    return new_state
