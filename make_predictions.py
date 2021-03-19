import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from logistic_regression import cost_function, predict, gradient_descent

def make_predictions_logreg(X, y, iterations = 1500, learning_rate = 0.01, plot_conv = False):
    """
    Make predictions using functions defined in logistic_regression.py

    :param X: Input data features. A matrix
    :param y: Input data labels. A vector
    :param iterations: Number of iterations training should go for
    :param learning_rate: Learning rate
    return
    """

    # Number of records
    m = len(y)

    # Add first column of 1's to the features matrix
    X = np.hstack((np.ones((m, 1)), X))

    # Number of features
    n = np.size(X, 1)

    # Initialize parameters
    theta = np.zeros((n, 1))

    # Initial cost
    initial_cost = cost_function(X = X, y = y, theta = theta)

    print("Initial cost: {}".format(initial_cost))

    print("--------------------------------------------------------------------------------------")
    print("Training the model...")

    # Training the model
    (cost_history, optimal_theta) = gradient_descent(X = X, y = y, theta = theta, learning_rate = learning_rate, num_iter = iterations)

    print("Training Finished.")
    print("Optimal parameters: {}".format(optimal_theta))

    # Take predictions
    preds = predict(X = X, theta = optimal_theta)

    # Calculate score
    accuracy = float(sum(preds == y)) / float(len(y))

    print("Accuracy: {}".format(accuracy))

    if plot_conv:
        plot_convergence(cost_history)

def plot_convergence(cost_history):
    """
    Plots convergence graph

    :param cost_history:
    return
    """

    plt.figure()
    sns.set_style('white')

    # Plot graph
    plt.plot(range(len(cost_history)), cost_history, 'r')

    plt.title("Convergence Graph")
    plt.xlabel("Number of Iterations Done")
    plt.ylabel("Cost")

    plt.show()