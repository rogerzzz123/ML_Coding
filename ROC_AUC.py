import numpy as np

def roc_auc(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    sorted_idx = np.argsort(-y_pred, kind='mergesort')
    y_true=y_true[sorted_idx]
    y_pred=y_pred[sorted_idx]

    # TURE POSITIVE AND FALSE POSITIVE 
    # TP = Corrected predicted TRUE/Actual TRUE
    # FP = Incorrected predicted TRUE/Actual FALSE

    tpr=[0]
    fpr=[0]
    tp=0
    fp=0


    actual_p=sum(y_true)
    actual_f=len(y_true)-actual_p

    if actual_p == 0 or actual_f == 0:  # Edge case: All positive or all negative
        return 1.0 if actual_p == len(y_true) else 0.0, tpr, fpr

    for i in range(len(y_true)):
        if y_true[i]==1:
            tp+=1
        else:
            fp+=1
        tpr.append(tp/actual_p)
        fpr.append(fp/actual_f) 
    
    auc_value=np.trapz(tpr, fpr)
    return auc_value, tpr, fpr

test_cases = [
    {
        "name": "Basic Case",
        "y_true": [0, 1, 1, 1, 0, 1, 0, 0, 1, 0],
        "y_pred": [0.1, 0.8, 0.6, 0.9, 0.2, 0.75, 0.3, 0.4, 0.65, 0.05],
        "expected": "Between 0 and 1"  # This is a standard case, should return valid AUC
    },
    {
        "name": "Perfect Prediction",
        "y_true": [0, 0, 0, 1, 1, 1],
        "y_pred": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
        "expected": 1.0  # Perfect ranking, should return AUC = 1.0
    },
    {
        "name": "Worst Prediction",
        "y_true": [0, 0, 0, 1, 1, 1],
        "y_pred": [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],
        "expected": 0.0  # Completely reversed, should return AUC = 0.0
    },
    {
        "name": "All Positives",
        "y_true": [1, 1, 1, 1, 1],
        "y_pred": [0.5, 0.6, 0.7, 0.8, 0.9],
        "expected": 1.0  # Since all are positives, should return 1.0
    },
    {
        "name": "All Negatives",
        "y_true": [0, 0, 0, 0, 0],
        "y_pred": [0.5, 0.6, 0.7, 0.8, 0.9],
        "expected": 0.0  # Since all are negatives, should return 0.0
    },
    {
        "name": "Tied Scores",
        "y_true": [1, 1, 1, 0, 1, 0, 1],
        "y_pred": [0.3, 0.6, 0.6, 0.4, 0.7, 0.4, 0.7],  # Multiple 0.4s, 0.6s, and 0.7s
        "expected": "Between 0 and 1"  # Should still be valid AUC
    },
    {
        "name": "Unbalanced Dataset (Many Negatives)",
        "y_true": [1, 0, 0, 0, 0, 0, 1, 1],
        "y_pred": [0.1, 0.2, 0.15, 0.3, 0.25, 0.35, 0.9, 0.85],
        "expected": "Between 0 and 1"  # Should still compute correctly
    },
    {
        "name": "Single Example (Positive)",
        "y_true": [1],
        "y_pred": [0.9],
        "expected": 1.0  # Only one point, should return AUC = 1.0
    },
    {
        "name": "Single Example (Negative)",
        "y_true": [0],
        "y_pred": [0.1],
        "expected": 0.0  # Only one point, should return AUC = 0.0
    }
]

# Run Tests
for case in test_cases:
    auc, tpr, fpr = roc_auc(case["y_true"], case["y_pred"])
    print(f"Test: {case['name']}, AUC: {auc:.4f}, Expected: {case['expected']}")
