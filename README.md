# Logarithmic Potential Activation Function (LPAF)
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
- [Performance Considerations](#performance-considerations)
- [Future Directions](#future-directions)
- [License](#license)

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
\[
f(x) = x \cdot \ln(1 + x^2)
\]

**Properties:**
- Non-linear scaling of the input.
- Intended to maintain significant gradients across a wide input range.
- Offers a unique interpretation by linking activation strength to a logarithmic potential function.

While LPAF was conceptually promising, empirical results on image classification tasks did not surpass standard baselines like ReLU.

---

### 2. Modified LPAF

**Goal of Modification:**
- Introduce scaling and bounding to reduce numerical instability.
- Improve gradient behavior and control output magnitude.

**Modified Definition (example modification):**
\[
f(x) = \tanh(\alpha \cdot x \cdot \ln(1 + x^2))
\]
with a small scaling factor \(\alpha\).

**Properties:**
- The use of `tanh` ensures outputs are bounded between -1 and 1.
- Scaling factor \(\alpha\) controls the function’s growth, preventing excessively large values.
- Improves stability compared to the original LPAF.

Although more stable than the original LPAF, this modified version still did not offer a clear performance advantage over established functions like ReLU in practical experiments.

---

### 3. Exponential Linear Squasher (ELS)

**Definition:**
\[
f(x) = 
\begin{cases} 
x & \text{if } x \geq 0 \\
x \cdot e^{x} & \text{if } x < 0
\end{cases}
\]

**Derivative:**
\[
f'(x) = 
\begin{cases} 
1 & \text{if } x \geq 0 \\
e^{x}(1 + x) & \text{if } x < 0
\end{cases}
\]

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



<img width="743" alt="image" src="https://github.com/user-attachments/assets/dbc608c0-7b6f-4694-b13b-cbbb64a9c80d">

# 
<img width="607" alt="image" src="https://github.com/user-attachments/assets/9b530098-e483-4266-9b75-2d65b7639522">

##
<img width="602" alt="image" src="https://github.com/user-attachments/assets/4e8c592c-9b30-4ef9-ab3b-ac7d17f4ecd5">

