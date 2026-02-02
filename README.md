# The Perceptron

**ECE 4367 – Pattern Recognition**  
Author: Alejandro Rubio

---

## Overview

This project introduces the **Perceptron**, one of the earliest and most fundamental algorithms in machine learning and pattern recognition. The perceptron is a linear binary classifier that assigns an input vector to one of two classes based on a learned decision boundary.

The perceptron output can be represented using either of the following label conventions:

$$y \in \{-1, +1\} \quad \text{or} \quad y \in \{0, 1\}$$

In practice, only one labeling convention is used consistently throughout training. In this project, both formulations are presented for completeness, but the learning rule is derived explicitly for the $\{0, 1\}$ case.

---

## Main Idea

The perceptron operates by computing a weighted sum of the input features, adding a bias term, and passing the result through a threshold-based activation function.

**Given:**
- input vector $\vec{x}$,
- weight vector $\vec{w}$,
- bias $w_0$,

the perceptron defines a decision function:

$$d(\vec{x}) = \vec{w}^T \vec{x} + w_0$$

The decision boundary $d(\vec{x}) = 0$ represents a hyperplane in $L$-dimensional space that separates the two classes.

Classification is performed as:

$$d(\vec{x}) \geq 0 \Rightarrow \vec{x} \in C_1$$
$$d(\vec{x}) < 0 \Rightarrow \vec{x} \in C_2$$

Here, the dimensionality $L$ corresponds to the number of features or measurements per data point.

---

## Input Vector $\vec{x}$

The input vector consists of the feature values describing a single data point:

$$\vec{x} = [x_1, x_2, \ldots, x_L]^T \in \mathbb{R}^L$$

A dataset composed of multiple samples can be represented as a matrix or tensor, where each row corresponds to one input vector.

All learning and classification behavior of the perceptron is driven entirely by the input data.

---

## Weight Vector $\vec{w}$ and Bias $w_0$

The weight vector $\vec{w}$ contains the parameters learned during training:

$$\vec{w} = [w_1, w_2, \ldots, w_L]^T \in \mathbb{R}^L$$

Each weight $w_i$ controls the influence of the corresponding input feature $x_i$ on the classification decision.

The bias term $w_0 \in \mathbb{R}$ shifts the decision boundary and allows the separating hyperplane to move away from the origin.

---

## Perceptron Definition

The perceptron computes the **net input** (activation) as:

$$z = \vec{w}^T \vec{x} + w_0 = \sum_{i=1}^{L} w_i x_i + w_0$$

An activation function is then applied to produce the output.

**$\{-1, +1\}$ formulation:**
$$y = f(z) = \begin{cases} +1 & \text{if } z \geq 0 \\ -1 & \text{if } z < 0 \end{cases}$$

**$\{0, 1\}$ formulation (used for learning here):**
$$y = f(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

---

## Augmented Vector Form

To simplify notation and implementation, the bias term can be absorbed into the weight vector using **augmented vectors**.

**Augmented weight vector:**

$$\vec{w}_{\text{aug}} = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_L \end{bmatrix} \in \mathbb{R}^{L+1}$$

**Augmented input vector:**

$$\vec{x}_{\text{aug}} = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_L \end{bmatrix} \in \mathbb{R}^{L+1}$$

Using this representation, the perceptron output is written compactly as:

$$y = \text{sign}(\vec{w}_{\text{aug}}^T \vec{x}_{\text{aug}})$$

This form treats the bias as an additional weight and is commonly used in implementations and theoretical analysis.

---

## The Learning Rule

The learning rule describes how the perceptron updates its parameters in response to misclassifications.

### Algorithm Overview

1. Initialize weights $\vec{w}$ and bias $w_0$ with small random values.
2. Choose a learning rate $\rho > 0$.
3. Iterate over the training dataset for multiple epochs.
4. For each training sample $(\vec{x}_i, t_i)$, perform the following steps.

#### Step 1: Compute Net Input
$$z_i = \vec{w}^T \vec{x}_i + w_0$$

#### Step 2: Apply Activation Function
$$y_i = \begin{cases} 1 & \text{if } z_i \geq 0 \\ 0 & \text{if } z_i < 0 \end{cases}$$

#### Step 3: Error Calculation

Let the true label be:

$$t_i \in \{0, 1\}$$

Define the error as:

$$E_i = t_i - y_i$$

This yields:
- $E_i = 0$: correct classification
- $E_i = 1$: false negative
- $E_i = -1$: false positive

The error term encodes both the need for an update and the direction of the update.

#### Step 4: Parameter Updates

Updates occur only when $E_i \neq 0$.

**Weight update:**
$$\vec{w}_{k+1} = \begin{cases} \vec{w}_k + \rho \, \vec{x}_i & \text{if } t_i = 1 \text{ and } y_i = 0 \\ \vec{w}_k - \rho \, \vec{x}_i & \text{if } t_i = 0 \text{ and } y_i = 1 \\ \vec{w}_k & \text{if } t_i = y_i \end{cases}$$

**Bias update:**
$$w_{0,k+1} = \begin{cases} w_{0,k} + \rho & \text{if } t_i = 1 \text{ and } y_i = 0 \\ w_{0,k} - \rho & \text{if } t_i = 0 \text{ and } y_i = 1 \\ w_{0,k} & \text{if } t_i = y_i \end{cases}$$

---

#### Geometric Definition of Perceptron


The perceptron can be understood geometrically as a method for learning a linear decision boundary that separates two classes in feature space.

##### Decision Boundary as a Hyperplane

Recall the decision function:

$$d(\vec{x}) = \vec{w}^T \vec{x} + w_0$$

The decision boundary is defined by:

$$d(\vec{x}) = 0$$

Geometrically, this equation represents a hyperplane in $\mathbb{R}^L$:
* In $\mathbb{R}^2$: a line
* In $\mathbb{R}^3$: a plane
* In $\mathbb{R}^L$: a hyperplane

The weight vector $\vec{w}$ is normal (perpendicular) to this hyperplane.

##### Role of the Weight Vector $\vec{w}$

The direction of $\vec{w}$ determines the orientation of the decision boundary.

* Points for which:
$$\vec{w}^T \vec{x} + w_0 > 0$$
lie on one side of the hyperplane

* Points for which:
$$\vec{w}^T \vec{x} + w_0 < 0$$
lie on the opposite side

Thus, classification depends on which side of the hyperplane a point lies on.

##### Role of the Bias $w_0$

The bias term $w_0$ controls the position of the hyperplane:
* Changing $w_0$ shifts the hyperplane parallel to itself
* Without a bias term, the hyperplane would be constrained to pass through the origin

##### Signed Distance Interpretation

The quantity:

$$\vec{w}^T \vec{x} + w_0$$

is proportional to the signed distance from a point $\vec{x}$ to the decision boundary.

* Sign → which side of the boundary the point lies on
* Magnitude → how far the point is from the boundary (up to a scaling factor)

This explains why the sign of $z$ alone is sufficient for classification.

##### Geometric Meaning of Misclassification

For a labeled training sample $(\vec{x}_i, t_i)$, an error occurs when the point lies on the wrong side of the decision boundary.

Using the $\{0,1\}$ labeling convention:
* $t_i = 1$ but $\vec{x}_i$ lies in the region $d(\vec{x}_i) < 0$
* $t_i = 0$ but $\vec{x}_i$ lies in the region $d(\vec{x}_i) \ge 0$

In either case, the hyperplane must be adjusted.

##### Geometric Interpretation of the Learning Rule

When a misclassification occurs, the perceptron updates its parameters as:

$$\vec{w} \leftarrow \vec{w} + \rho E_i \vec{x}_i$$
$$w_0 \leftarrow w_0 + \rho E_i$$

Geometrically:
* If $t_i = 1$ and the point is misclassified, the update moves the hyperplane toward the positive example
* If $t_i = 0$ and the point is misclassified, the update moves the hyperplane away from the negative example

This adjustment:
* Rotates the hyperplane (via $\vec{w}$)
* Translates the hyperplane (via $w_0$)

Each update reduces the classification error for the current sample.

### Key Geometric Insight

The perceptron learns by iteratively rotating and shifting a hyperplane until as much of the  training points lie on the correct side.

If the data is linearly separable, this process is guaranteed to converge in a finite number of updates. - Perceptron Convergence Theorem

---

## Summary

- The perceptron is a **mistake-driven** learning algorithm
- Updates occur only for **misclassified** samples
- The decision boundary is a **linear hyperplane**
- Learning is **guaranteed to converge** if the data is linearly separable

This implementation demonstrates the foundational principles of linear classification and serves as a stepping stone toward more advanced models in pattern recognition.