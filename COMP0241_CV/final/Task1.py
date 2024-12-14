import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# Task1.a -- Algorithm 1: Region Extraction Based on Color Thresholding and Morphological Operations
def process_image(image_path):
    """
    Process an image to extract regions matching a specific HSV color range and visualize the results.

    Args:
        image_path (str): Path to the input image.
    Returns:
        None
    """

    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_hsv = (80, 20, 50)  
    upper_hsv = (130, 255, 255) 


    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(mask, 50, 150)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    binary_mask = np.zeros_like(mask)

    if contours:
        contour = max(contours, key=cv2.contourArea)

        if len(contour) >= 5: 
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(binary_mask, ellipse, 255, -1)  
        else:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(binary_mask, center, radius, 255, -1)

    white_background = np.ones_like(image) * 255

    extracted_region = np.where(binary_mask[..., None] == 255, image, white_background)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Extracted Region")
    plt.imshow(cv2.cvtColor(extracted_region, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    overlay = image.copy()
    overlay[binary_mask == 255] = (255, 255, 165)

    alpha = 0.5
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Original + Mask")
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Task1.a -- Algorithm 2: Circular Detection Using Hough Transform
def process_hough_circle(image_path):
    """
    Process an image to detect the largest circle using Hough Transform, generate a binary mask,
    and visualize the original image, mask, and extracted region.

    Args:
        image_path (str): Path to the input image.

    Returns:
        None
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)

    # edges = cv2.Canny(gray, 100, 150)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=500,
                               param1=100, param2=20, minRadius=200, maxRadius=300)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])  

        mask = np.zeros_like(gray)
        cv2.circle(mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, thickness=-1)
    else:
        mask = np.zeros_like(gray)

    white_background = np.ones_like(image) * 255  
    extracted_region = np.where(mask[..., None] == 255, image, white_background) 

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Binary Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Extracted Region")
    plt.imshow(cv2.cvtColor(extracted_region, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    overlay = image.copy()
    overlay[mask == 255] = (255, 255, 165)

    alpha = 0.5 
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Original + Mask")
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Task1.b 
def detect_largest_circle(image_path, min_dist=300, param1=150, param2=0.01, min_radius=0, max_radius=0, display_results=True):
    """
    Detects the largest circle in an image and returns its binary mask.
    
    Args:
        image_path (str): Path to the image file.
        min_dist (int): Minimum distance between the centers of detected circles.
        param1 (int): First method-specific parameter for cv2.HoughCircles.
        param2 (int): Second method-specific parameter for cv2.HoughCircles.
        min_radius (int): Minimum circle radius.
        max_radius (int): Maximum circle radius.
        display_results (bool): Whether to visualize the results.
    
    Returns:
        np.ndarray: Binary mask of the largest detected circle.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found at the provided path.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])  
        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, thickness=-1)

        if display_results:
            cv2.circle(image, (largest_circle[0], largest_circle[1]), largest_circle[2], (0, 255, 0), 4)
            cv2.circle(image, (largest_circle[0], largest_circle[1]), 2, (0, 0, 255), 3)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Detected Circle")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("Binary Mask of Circle")
            plt.imshow(mask, cmap="gray")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        return mask
    else:
        if display_results:
            print("No circles detected.")
        return np.zeros_like(gray, dtype=np.uint8) 
    
# Task1.c 
def calculate_roc_curve():
    """
    Calculated ROC curve
    """
    roc_points = []
    for i in range(100):
        image_path = f"dataset/images/{i:06d}.png"
        ground_truth_path = f"dataset/masks/{i:06d}.png"
        predicted_mask = detect_largest_circle(image_path)

        ground_truth_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        print(f"The predicted_mask_image_files: {image_path}")
        print(f"The ground_truth_mask_image_files: {ground_truth_path}")

        _, predicted_mask = cv2.threshold(predicted_mask, 127, 1, cv2.THRESH_BINARY)
        _, ground_truth_mask = cv2.threshold(ground_truth_mask, 127, 1, cv2.THRESH_BINARY)

        # Calculate ROC points
        fpr, tpr = calculate_roc_points(predicted_mask, ground_truth_mask)

        roc_points.append((fpr, tpr))

    fpr_values, tpr_values = zip(*roc_points)

    plt.figure()
    plt.scatter(fpr_values, tpr_values, color='orange', label=f'ROC Points (Total: {len(roc_points)})', s=50, edgecolor='black', alpha=0.8)
    plt.plot([0, 1], [0, 1], 'b--', label='Random Chance')
    plt.title('ROC Points for AO Segmentation')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.show()

# Task1.c -- Function to calculate true positive rate and false positive rate manually
def calculate_roc_points(predicted_mask, ground_truth_mask):
    """
    Calculated ROC points
    
    Args:
        predicted_mask
        ground_truth_mask
    
    Returns:
        tuple: A tuple (FPR, TPR), where:
            - FPR: False Positive Rate
            - TPR: True Positive Rate
    """
    predicted_flat = predicted_mask.flatten()
    ground_truth_flat = ground_truth_mask.flatten()

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives
    TN = 0  # True Negatives

    # Calculate TP, FP, FN, TN
    for pred, gt in zip(predicted_flat, ground_truth_flat):
        if gt == 1 and pred == 1:
            TP += 1
        elif gt == 0 and pred == 1:
            FP += 1
        elif gt == 1 and pred == 0:
            FN += 1
        elif gt == 0 and pred == 0:
            TN += 1

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  

    return FPR, TPR