import numpy as np

def mse(y_true, y_pred):
    """
    Compute Mean Squared Error (MSE) loss.

    Parameters:
    y_true (numpy array): Ground truth values
    y_pred (numpy array): Predicted values

    Returns:
    float: MSE loss
    """
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([3.0, -0.5, 2.0, 7.0])
y_pred = np.array([2.5, 0.0, 2.0, 8.0])

print("MSE Loss:", mse(y_true, y_pred))  # Output: 0.375