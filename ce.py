# Cross Entropy

## numpy implementation
import numpy as np
def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred=np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true*np.log(y_pred))/y_true.shape[0]


def cross_entropy(y_true, y_pred):
    eps=1e-12
    y_pred=np.clip(y_pred, eps, 1-eps) # make it between eps and 1-eps
    return -np.sum(y_true * np.log(y_pred))/y_true.shape[0]

y_true = np.array([[1, 0, 0],  # Class 0
                   [0, 1, 0],  # Class 1
                   [0, 0, 1]]) # Class 2

y_pred = np.array([[0.7, 0.2, 0.1],  # Predicted probs for class 0
                   [0.1, 0.8, 0.1],  # Predicted probs for class 1
                   [0.2, 0.3, 0.5]]) # Predicted probs for class 2

print("Cross-Entropy Loss:", cross_entropy(y_true, y_pred))  # Output: ~0.366


## pytorch implementation

## torch cross entropy loss takes logit!!


import torch
import torch.nn as nn

ce_loss=nn.CrossEntropyLoss()

# Example inputs
y_true_torch = torch.tensor([0, 1, 2])  # Class indices (not one-hot!)
y_pred_torch = torch.tensor([[0.7, 0.2, 0.1], 
                             [0.1, 0.8, 0.1], 
                             [0.2, 0.3, 0.5]])
y_pred_softmax = torch.tensor([[0.7, 0.2, 0.1],  
                               [0.1, 0.8, 0.1],  
                               [0.2, 0.3, 0.5]])

# Convert softmax probabilities back to logits
y_pred_logits = torch.log(y_pred_softmax)  # Log probabilities
y_pred_logits_shifted = torch.log(y_pred_softmax) - torch.logsumexp(torch.log(y_pred_softmax), dim=1, keepdim=True)

# Compute loss
loss = ce_loss(y_pred_logits_shifted, y_true_torch)
print("Cross-Entropy Loss (PyTorch):", loss.item())  
