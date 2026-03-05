import numpy as np

class LinearRegression:
    """
    Linear Regression Class implemented using Batch Gradient Descent.

    Parameters:
        learning_rate: step size for gradient updates.
        n_epochs: number of training epochs.
    """

    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None #Initialized to array of 0s when .fit is called.
        self.bias = None #Initialized to 0 when .fit is called.

    def fit(self, X, y):
        """
        Trains the Linear Regression model using Batch Gradient Descent. 

        Args:
            X: 2D input array (n_samples, n_features) shape.
            y: 1D output array (n_samples,) shape.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_epochs):
            y_predicted = np.dot(X, self.weights) + self.bias #1D array of predicted values (n_samples,) shape.

            #X.T shape is (n_features, n_samples)
            #(y_predicted - y) is (n_samples,) shape
            #dot product results in (n_features,) shape
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y)) #1D array of gradients to update weights (n_features,) shape.
            db = (1/n_samples) * np.sum((y_predicted - y))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts output using the trained model 
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted