"""
This module contains RxR -> R functions to test the optimizers on
"""

import numpy as np


def f1(state, derivative=False):
    if derivative:
        #TODO: evaluate derivative of function in state
        pass
    else:
        #TODO: evaluate function in state
        pass


def sphere(state, derivative=False):
    if derivative:
        return 2 * state
    else:
        return np.dot(state, state)
