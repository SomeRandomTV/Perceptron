# Perceptron Class

import numpy as np


class Perceptron:
    """
    A simple perceptron classifier for binary classification.

    The perceptron learns a linear decision boundary by iteratively
    adjusting weights based on prediction errors.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 1.0):
        """
        Initialize the perceptron with training data.

        Args:
            x: Input features as a 2D numpy array (samples x features)
            y: Target labels as a 1D numpy array
            learning_rate: Step size for weight updates (default: 1.0)
        """
        self.input_features: np.ndarray = x
        self.learning_rate: float = learning_rate
        self.num_samples, self.num_features = x.shape
        self.weights: np.ndarray = np.zeros(self.num_features)
        self.bias: float = 0.0
        self.labels: np.ndarray = y

        self.epochs: int = 40
        self.iterations = self.num_samples

    @staticmethod
    def _compute_weighted_sum(
        input_features: np.ndarray, weights: np.ndarray, bias: float
    ):
        """
        Calculate the weighted sum of inputs plus bias.

        Returns:
            numpy array: Weighted sum for all samples
        """
        weighted_sum = np.dot(input_features, weights) + bias

        return weighted_sum

    @staticmethod
    def _activation_function(weighted_sum) -> int:
        """
        Apply step activation function to weighted sum.

        Args:
            weighted_sum: The weighted sum input

        Returns:
            int: 0 if weighted_sum < 0, else 1
        """
        if weighted_sum < 0:
            return 0

        return 1
    
    @staticmethod
    def _calculate_error(prediction: int, truth_label):

        error = truth_label - prediction

        return error


    def fit(self):
        """
        Train the perceptron on the input data.

        Iterates through epochs, making predictions and updating weights
        based on prediction errors using the perceptron learning rule.
        """
        rho = self.learning_rate
        features = self.input_features
        truth_labels = self.labels
        weights = self.weights
        bias = self.bias

        for epoch in range(self.epochs):
            
            print(f"On training epoch: {epoch}")
            for i in range(self.num_samples):

                x_i = features[i]
                t_i = truth_labels[i]
                
                
                weighted_sum = self._compute_weighted_sum(
                    input_features=x_i, weights=weights, bias=bias
                )

                activation = self._activation_function(weighted_sum=weighted_sum)
                
                err = self._calculate_error(prediction=activation, truth_label=t_i)
                
                # update weights
                weights = weights + rho * err * x_i
                
                # update bias
                bias = bias + rho * err
                
                
            self.weights = weights
            self.bias = bias
            

    def predict(self, new_data: np.ndarray):
        """
        Make predictions on new data.

        Args:
            new_data: Input features as a 2D numpy array (samples x features)

        Returns:
            numpy array: Predicted labels (0 or 1)
        """
        weighted_sum = self._compute_weighted_sum(new_data, self.weights, self.bias)
        predictions = np.array([self._activation_function(ws) for ws in weighted_sum])
        return predictions
    
