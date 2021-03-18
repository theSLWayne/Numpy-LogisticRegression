import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import seaborn as sns

def sigmoid(x):
    """
    Implementation of sigmoid function. 
    Always outputs values in the range of 0-1 
    hence used to calculate probability of a input data instance belonging to a certain class.

    This function can be applied to all elements of a numpy array at once
    because exponential function from Numpy is used

    :param x:
    return
    """

    return 1 / (1 + np.exp(-x))

