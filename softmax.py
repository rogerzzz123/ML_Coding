import numpy as np

def softmax(x):
    exp_x=np.exp(x-np.max(x)) # safe softmax
    return exp_x/np.sum(exp_x, axis=-1, keepdims=True)
## axis=-1 applies np.sum() along the last axis of the array.

## Ensures the shape remains compatible for broadcasting.

x = np.array([2.0, 1.0, 0.1])
print(softmax(x))  # Output: [0.65900114 0.24243297 0.09856589]