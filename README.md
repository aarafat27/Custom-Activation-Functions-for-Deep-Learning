# Custom Activation Functions for Deep Learning

This repository introduces three custom activation functions designed to address various challenges in training deep neural networks, especially for image classification tasks. The functions are:

1. **Logarithmic Potential Activation Function (LPAF)**
2. **Modified Logarithmic Potential Activation Function (Modified LPAF)**
3. **Exponential Linear Squasher (ELS)**

Each activation function aims to tackle issues such as the vanishing gradient problem, improve interpretability, maintain computational efficiency, and ultimately enhance the learning capacity of neural networks.

---

## Table of Contents
- [Introduction](#introduction)
- [Activation Functions](#activation-functions)
  - [1. Logarithmic Potential Activation Function (LPAF)](#1-logarithmic-potential-activation-function-lpaf)
  - [2. Modified LPAF](#2-modified-lpaf)
  - [3. Exponential Linear Squasher (ELS)](#3-exponential-linear-squasher-els)
- [Key Features and Advantages](#key-features-and-advantages)
- [Usage](#usage)

---

## Introduction

Traditional activation functions such as ReLU, Sigmoid, and Tanh have known limitations, including vanishing or exploding gradients and the risk of "dead neurons." To address these challenges, researchers and practitioners have explored alternative activation functions.

This repository presents three custom activations—LPAF, Modified LPAF, and ELS—developed to:
- Provide non-linearity essential for deep models.
- Mitigate the vanishing and exploding gradient issues.
- Offer interpretability and smooth gradient flow.
- Enhance training stability and potentially improve model performance.

---

## Activation Functions

### 1. Logarithmic Potential Activation Function (LPAF)

**Definition:**

<img width="482" alt="image" src="https://github.com/user-attachments/assets/537a1979-3a4e-4b85-b52e-2b516e23cf72">

**Derivative:**

<img width="497" alt="image" src="https://github.com/user-attachments/assets/f3f895a1-31f6-4e4c-9cb5-97085370f409">


**Properties:**
- Non-linear scaling of the input.
- Intended to maintain significant gradients across a wide input range.
- Offers a unique interpretation by linking activation strength to a logarithmic potential function.



---

### 2. Modified LPAF

**Goal of Modification:**
- Introduce scaling and bounding to reduce numerical instability.
- Improve gradient behavior and control output magnitude.

**Modified Definition (example modification):**

<img width="514" alt="image" src="https://github.com/user-attachments/assets/8436d967-3208-45d2-a6a4-9e5e0fc5a714">


**Properties:**
- The use of `tanh` ensures outputs are bounded between -1 and 1.
- Scaling factor \(\alpha\) controls the function’s growth, preventing excessively large values.
- Improves stability compared to the original LPAF.

---

### 3. Exponential Linear Squasher (ELS)

**Definition:**

<img width="422" alt="image" src="https://github.com/user-attachments/assets/2951deae-ab70-48de-84da-9fe4514cb174">


**Derivative:**

<img width="447" alt="image" src="https://github.com/user-attachments/assets/32e248ae-7cdf-4b59-8053-6ae952e0924d">


**Properties:**
- Smoothly transitions between linear behavior for positive inputs and a smoothly decaying exponential for negative inputs.
- Ensures non-zero gradients for negative inputs, preventing "dead neurons."
- Maintains stable gradients and is computationally efficient.

ELS is designed to combine the simplicity and efficiency of ReLU with improved handling of negative values, offering a potentially more robust alternative.

---

## Key Features and Advantages

- **Non-Linearity:** All three functions introduce non-linear transformations to capture complex data patterns.
- **Gradient Stability:** Designed to reduce the risk of vanishing or exploding gradients.
- **Interpretability:** Outputs reflect meaningful transformations of inputs, aiding in understanding model behavior.
- **Computational Efficiency:** Functions rely on operations (multiplications, exponentials, logarithms) that are efficiently computed on modern hardware.
- **Compatibility:** Easily integrable into existing deep learning frameworks (e.g., TensorFlow, PyTorch) by defining custom layers.

---

## Usage

To use these activation functions, simply integrate them into your neural network code as custom layers or activation functions. For example, in TensorFlow/Keras:

```python
import tensorflow as tf

def lpaf(x):
    return x * tf.math.log(1 + x**2)

def modified_lpaf(x):
    alpha = 0.1
    return tf.math.tanh(alpha * x * tf.math.log(1 + x**2))

def els(x):
    return tf.where(x >= 0, x, x * tf.math.exp(x))

# Example usage in a Keras model:
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(32,32,3)),
    tf.keras.layers.Lambda(els),  # Using ELS activation
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
  ```


