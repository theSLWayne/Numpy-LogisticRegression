from sklearn.datasets import make_classification
import numpy as np

from make_predictions import make_predictions_logreg

# Create dataset belonging to two categories
X, y = make_classification(n_samples = 5000, n_features = 2, n_redundant = 0, n_informative = 1,
                             n_clusters_per_class = 1, random_state = 44)

y = y[:, np.newaxis]

# Make predictions
make_predictions_logreg(X = X, y = y, iterations = 2000, learning_rate = 0.3, plot_conv = True)