# Perceptron Class

import numpy as np
from typing import Tuple

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
     
    
    
    def compute_weighted_sum(self):
        """
        Calculate the weighted sum of inputs plus bias.
        
        Returns:
            numpy array: Weighted sum for all samples
        """
        weighted_sum = np.dot(self.input_features, self.weights) + self.bias
        
        return weighted_sum
    
    
    def activation_function(self, weighted_sum) -> int:
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
    
    def fit(self):
        """
        Train the perceptron on the input data.
        
        Iterates through epochs, making predictions and updating weights
        based on prediction errors using the perceptron learning rule.
        """
        pass
    
    def predict(self, new_data: np.ndarray):
        """
        Make predictions on new data.
        
        Args:
            new_data: Input features as a 2D numpy array (samples x features)
            
        Returns:
            numpy array: Predicted labels (0 or 1)
        """
        pass
        
        
        
        
        
