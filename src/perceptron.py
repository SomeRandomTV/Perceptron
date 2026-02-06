# Perceptron Class

import numpy as np
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Perceptron:
    """
    A simple perceptron classifier for binary classification.

    The perceptron learns a linear decision boundary by iteratively
    adjusting weights based on prediction errors.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> None:
        """
        Initialize the perceptron with training data.

        Args:
            x: Input features as a 2D numpy array (samples x features)
            y: Target labels as a 1D numpy array
            learning_rate: Step size for weight updates (default: 0.01)
        """
        self.input_features: np.ndarray = x
        self.learning_rate: float = learning_rate
        self.num_samples, self.num_features = x.shape
        self.weights: np.ndarray = np.random.randn(self.num_features) * 0.01
        self.bias: float = 0.0
        self.labels: np.ndarray = y

        self.epochs: int = 100
        self.iterations = self.num_samples


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
    def _calculate_weighted_sum(input_data: np.ndarray, weights: np.ndarray, bias):
        
        
        weighted_sum = np.dot(input_data, weights) + bias
        
        return weighted_sum
        
        
        
    
    @staticmethod
    def _calculate_error(prediction: int, truth_label) -> int:

        error = truth_label - prediction

        return error


    def fit(self) -> int:
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
            
            misclassifications: int = 0
            print(f"========== Epoch: {epoch} =========")
            for i in range(self.num_samples):
                
                x_i = features[i]
                t_i = truth_labels[i]
                
                weighted_sum = np.dot(x_i, weights) + bias

                activation = self._activation_function(weighted_sum=weighted_sum)
                err = self._calculate_error(prediction=activation, truth_label=t_i)
                if err != 0:
                    misclassifications += 1
                    
                # update weights
                weights = weights + rho * err * x_i
                # print(f"-- Weights: {weights}")
                # update bias
                bias = bias + rho * err
                # print(f"-- Bias: {bias}")
                # print(f"-- Misclassifications: {misclassifications}")
                
            self.weights = weights
            self.bias = bias
            self._plot_decision_boundary(epoch=epoch)
            
            print(f"======= Weights + Bias after Epoch {epoch} ===========")
            print(f" -- Weights: {self.weights}")
            print(f" -- Bias: {self.bias}")
            
            if misclassifications == 0:
                print(f"Convergence achieved at epoch {epoch}")
                print(f"Misclassifications: {misclassifications}")
                return 0
            
        return 0    
            
            
    def _plot_decision_boundary(self, epoch: int) -> None:
        """
        Plot and display the decision boundary for the current epoch.

        Args:
            epoch: Current epoch number
        """
        X = self.input_features
        y = self.labels
        
        # separate classes for plotting
        class_0 = X[y == 0]
        class_1 = X[y == 1]
        
        plt.figure(figsize=(10, 8))
        
        # Plot training data
        plt.scatter(class_0[:, 0], class_0[:, 1], label='Class 0', alpha=0.6)
        plt.scatter(class_1[:, 0], class_1[:, 1], label='Class 1', alpha=0.6)
        
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx = np.linspace(x_min, x_max, 100)
        if self.weights[1] != 0:
            yy = -(self.weights[0] * xx + self.bias) / self.weights[1]
            plt.plot(xx, yy, 'k-', label='Decision Boundary', linewidth=2)
        
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'Perceptron Decision Boundary - Epoch {epoch}')
        plt.legend()
        plt.grid()
        plt.show()

    def predict(self, new_data: np.ndarray) -> int:
        """
        Make predictions on new data.

        Args:
            new_data: Input features as a 2D numpy array (samples x features)

        Returns:
            numpy array: Predicted labels (0 or 1)
        """
        weighted_sum = self._calculate_weighted_sum(input_data=new_data, weights=self.weights, bias=self.bias)
        prediction = self._activation_function(weighted_sum=weighted_sum)
        return prediction
    
    

