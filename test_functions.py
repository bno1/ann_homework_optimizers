"""
This module contains RxR -> R functions to test the optimizers on
"""

import numpy as np


def sphere(state, derivative=False):
    if derivative:
        return 2 * state
    else:
        return np.dot(state, state)


# Recommended range: [-1; 1]x[-1; 1]
# minimum at (1, 1) (== (A, A^2))
def rosenbrock(state, derivative=False):
    A = 1.0
    B = 50

    x = state[0]
    y = state[1]

    if derivative:
        return np.array((
            -2 * (A - x) - 4 * x * B * (y - x * x),
            2 * B * (y - x * x)
        ))
    else:
        return (A - x) ** 2 + B * (y - x * x) ** 2


# Recommended range: [-2; 2]x[-2; 2]
# minimum at (0, 0)
def rastrigin(state, derivative=False):
    A = 10.0
    TWO_PI = np.pi * 2

    if derivative:
        return 2 * state + A * np.sin(state * TWO_PI) * TWO_PI
    else:
        return A * 2.0 + (state * state - A * np.cos(state * TWO_PI)).sum()


# This function is just horrible
# minimum at (0, 0)
def griewank(state, derivative=False):
    F = 1.0 / 4000.0
    CF = np.array((1, np.sqrt(2) / 2))

    if derivative:
        x = state[0]
        y = state[1]

        return np.array((
            2 * F * x + np.sin(x) * np.cos(CF[1] * y),
            2 * F * y + np.sin(CF[1] * y) * CF[1] * np.cos(x)
        ))
    else:
        return 1 + F * (state * state).sum() - np.cos(state * CF).prod()


def saddle(state, derivative=False):
    if derivative:
        return 2 * state * [1.0, -1.0]
    else:
        return np.dot(state * state, [1.0, -1.0])
