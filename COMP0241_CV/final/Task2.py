import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt

from functions import detect_largest_circle


def Task2a(datasets, max_len=-1, display_results=True):
    """
    Determine the Geometric Centre in Images.

    Calculate the centroid of the AO using the image processing
    techniques you used in Task 1 (in pixel coordinate).

    Args:
        datasets List[{
            "path": str,
            "images": List[Dict{
                "name": str,
                "frame": int,
                "fps": float,
                "timestamp": int,
            }]
        }]: A list of folder dicts, each has keys "path" and "images",
            represent the folder path and a list of image dicts.

    Returns:
        Dict{
            path (str): {
                "centers": ndarray,
                "masks": ndarray,
            }
        }: The circle center, radius and masks for each image.
    """
    circles = {}
    radiusThreshold = 100
    distThreshold = 100
    alpha = 0.9

    for dataset in datasets:
        dataset = copy.copy(dataset)
        if 0 < max_len < len(dataset["images"]):
            dataset["images"] = dataset["images"][:max_len]

        circleList = []
        maskList = []
        historyCircle = [0, 0, 0]
        for imageDict in dataset["images"]:
            imagePath = imageDict["path"]
            image = cv2.imread(imagePath)
            circle = detect_largest_circle(image, min_dist=100, param1=50, param2=0.6, display_results=display_results)

            if circle is None:
                circleList.append(historyCircle)
                continue

            if (
                    (np.abs(historyCircle[2] - circle[2]) > radiusThreshold) or
                    (np.abs(historyCircle[0] - circle[0]) + np.abs(historyCircle[1] - circle[1]) > distThreshold)
                ) and not historyCircle[2] == 0:
                circle = historyCircle
            historyCircle = circle
            mask = np.zeros_like(image[:, :, 0])
            cv2.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)
            circleList.append(historyCircle)
            maskList.append(mask)

            if display_results:
                cv2.circle(image, (historyCircle[0], historyCircle[1]), 4, (0, 0, 255), 3)

                plt.title("Geometric Centre")
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.show()
        circles[dataset["path"]] = {
            "centers": np.array(circleList),  # [N, 3]
            "masks": np.array(maskList)
        }
    return circles


def Task2b(circles):
    """
    Assess the Movement of the Centre Over Time.

    Analyse whether the AO's centre shifts due to swinging and
    quantify this movement with the static camera (in pixel
    coordinate).

    Args:
        circles: Dict{
                path (str): {
                    "centers": ndarray,
                    "masks": ndarray,
                }
            }
    """
    for path, circle in circles.items():
        center = circle["centers"]
        cX = center[:, 0]
        cY = center[:, 1]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('x')
        plt.plot(cX)

        plt.subplot(1, 3, 2)
        plt.title('y')
        plt.plot(cY)

        plt.subplot(1, 3, 3)
        plt.title('traj')
        plt.plot(cX, cY)

        plt.tight_layout()
        plt.show()

        print(f"X: max {np.max(cX)}, min {np.min(cX)}, amplitude {np.max(cX) - np.min(cX)}")
        print(f"Y: max {np.max(cY)}, min {np.min(cY)}, amplitude {np.max(cY) - np.min(cY)}")
