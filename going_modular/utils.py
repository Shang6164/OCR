import numpy as np

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate precision, recall, and F1-score for binary predictions.

    Args:
        predictions (np.ndarray): Predicted values (e.g., heatmaps).
        targets (np.ndarray): Ground truth values.
        threshold (float): Threshold to binarize predictions.

    Returns:
        dict: A dictionary containing precision, recall, and F1-score.
    """
    predictions = (predictions > threshold).astype(int)
    targets = targets.astype(int)

    true_positive = np.sum((predictions == 1) & (targets == 1))
    false_positive = np.sum((predictions == 1) & (targets == 0))
    false_negative = np.sum((predictions == 0) & (targets == 1))

    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def visualize_heatmap(image, heatmap):
    """
    Overlay a heatmap on an image for visualization.

    Args:
        image (np.ndarray): The original image.
        heatmap (np.ndarray): The heatmap to overlay.

    Returns:
        np.ndarray: The image with the heatmap overlay.
    """
    import cv2
    heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay