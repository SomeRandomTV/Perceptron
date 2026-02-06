import numpy as np
from src.perceptron import Perceptron

# Perceptron Algorithm Implementation 
# ECE 4363 - Pattern Recognition
# Author: Alejandro Rubio
# Date: 02/02/2026

# Description:
#      The Perceptron is a linear binary classifier that assigns an input vector to one of two classes based on a learned decision boundary.
#      The decision boundary is given by d(x) = w * x + w_0 = 0
#      where:
#          w => Weight vector(Found in training)
#          x => Input Vector(Given data)
#          w_0 => Bias term

         
    

def main():
    """
    Main function to demonstrate the perceptron classifier.
    
    Creates linearly separable data, trains the perceptron,
    and visualizes the results.
    """
    
    # LINEARLY SEPARABLE DATA
    # Class 0: points around (-2, -2)
    class_0 = np.random.randn(100, 2) + np.array([-1, 2])
    # Class 1: points around (2, 2)
    class_1 = np.random.randn(100, 2) + np.array([-2, -3])
    
    # Combine and create labels
    X = np.vstack([class_0, class_1])
    Y = np.hstack([np.zeros(len(class_0)), np.ones(len(class_1))]).astype(int)
    
    
    print("===============================")
    print("Training Data Points:")
    for i, (point, label) in enumerate(zip(X, Y)):
        print(f"Point {i}: Features={point}, Label={label}")
    print("===============================")
    
    perc = Perceptron(X, Y, learning_rate=0.01)
    perc.fit()
    
if __name__ == "__main__":
    main()