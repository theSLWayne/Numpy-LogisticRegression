import numpy as np


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

def cost_function(X, y, theta):
    """
    Cost functino of logistic regression. Uses standard logistic regression loss function.

    :param X: Features of input data - a matrix
    :param y: Labels of input data - a vector
    :param theta: Parameters - a vector
    return
    """

    # Get number of records
    m = len(y)

    # Get hypothesis value using Sigmoid function. 
    # A vector resulting from matrix multiplication between X and theta will be passed to sigmoid function.
    h = sigmoid(X @ theta)

    # Calculate cost - .T is a python matrix operation term. Indicates transpose of a matrix
    cost = (-1 / m) * ((y.T @ np.log(h)) + (1 - y).T @ np.log(1 - h))

    return cost

def gradient_descent(X, y, theta, learning_rate, num_iter):
    """
    Perform gradient descent.

    :param X: Features of input data - a matrix
    :param y: Labels of input data - a vector
    :param theta: Parameters - a vector
    :param learning_rate: Learning rate
    :param num_iter: Number of iterations gradient descent should be performed
    return
    """

    # Number of records
    m = len(y)

    # Cost history - costs per each steps are recorded
    cost_history = np.zeros((num_iter, 1))

    # Gradient descent process
    for i in range(num_iter):
        # Update parameters
        theta -= (learning_rate / m) * (X.T @ (sigmoid(X @ theta) - y))

        # Record cost history
        cost_history[i] = cost_function(X, y, theta)

    return (cost_history, theta)

