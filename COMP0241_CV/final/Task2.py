import copy
import os.path

import cv2
import numpy as np
import matplotlib.pyplot as plt

from COMP0241_CV.final.utils import dataset_to_image_pair
from functions import detect_largest_circle, kalman_filter


def Task2a(datasets, imageReader, display_results=True):
    """
    Determine the Geometric Centre in Images.

    Calculate the centroid of the AO using the image processing
    techniques you used in Task 1 (in pixel coordinate).

    Args:
        datasets (List[Dict{
            "path": str,
            "images": List[Dict{
                "path": str,
                "frame": int,
                "fps": float,
                "timestamp": int,
            }]
        }]): A list of folder dicts, each has keys "path" and "images",
            represent the folder path and a list of image dicts.
        imageReader (ImageReader)
        display_results (bool)

    Returns:
        Dict{
            path (str): {
                "centers": ndarray,
                "masks": ndarray,
            }
        }: The circle center, radius and masks for each image.
    """
    circles = {}
    radiusThreshold = 10
    distThreshold = 35

    for dataset in datasets:
        circleList = []
        maskList = []
        historyCircle = [0, 0, 0]
        for imageDict in dataset["images"]:
            imagePath = imageDict["path"]
            image = imageReader.read_image_with_calibration(imagePath)
            # circle = detect_largest_circle(image, min_dist=100, param1=50, param2=0.6, display_results=False)
            circle = detect_largest_circle(image, min_dist=300, param1=150, param2=0.01, display_results=False)

            if circle is None:
                circle = historyCircle

            if (
                    (np.abs(historyCircle[2] - circle[2]) > radiusThreshold) or
                    (np.abs(historyCircle[0] - circle[0]) + np.abs(historyCircle[1] - circle[1]) > distThreshold)
                ) and not historyCircle[2] == 0:
                circle = historyCircle
            historyCircle = circle
            mask = np.zeros_like(image[:, :, 0])
            cv2.circle(mask, (historyCircle[0], historyCircle[1]), historyCircle[2], 255, thickness=-1)
            mask[mask == 255] = 1
            circleList.append(historyCircle)
            maskList.append(mask)

            if display_results:
                cv2.circle(image, (historyCircle[0], historyCircle[1]), 4, (0, 0, 255), 3)

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

        circles[dataset["path"]] = {
            "centers": np.array(circleList),  # [N, 3]
            "masks": np.array(maskList)
        }
    return circles


def Task2b(circles, display_results=True):
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
        display_results (bool)

    Returns:
        Dict{
            path (str): {
                "centers": ndarray,
                "masks": ndarray,
            }
        }
    """
    circles = copy.copy(circles)
    for path, circle in circles.items():
        center = circle["centers"]
        cX = center[:, 0]
        cY = center[:, 1]
        cR = center[:, 2]

        A = 1
        H = 1
        Q = 1e-5
        R = 1e-3
        P0 = 0.3

        ncX = kalman_filter(cX, A, H, Q, R, P0)
        ncY = kalman_filter(cY, A, H, Q, R, P0)
        ncR = kalman_filter(cR, A, H, Q, R, P0)

        ncX = kalman_filter(ncX, A, H, Q, R, P0)
        ncY = kalman_filter(ncY, A, H, Q, R, P0)
        ncR = kalman_filter(ncR, A, H, Q, R, P0)

        if display_results:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.title('x')
            plt.plot(cX)
            plt.plot(ncX)

            plt.subplot(1, 3, 2)
            plt.title('y')
            plt.plot(cY)
            plt.plot(ncY)

            plt.subplot(1, 3, 3)
            plt.title('r')
            plt.plot(cR)
            plt.plot(ncR)
            plt.show()

            plt.title('traj')
            plt.plot(ncX, ncY)
            plt.show()

        print(f"X: max {np.max(cX)}, min {np.min(cX)}, amplitude {np.max(cX) - np.min(cX)}")
        print(f"Y: max {np.max(cY)}, min {np.min(cY)}, amplitude {np.max(cY) - np.min(cY)}")

        circles[path]["centers"][:, 0] = ncX
        circles[path]["centers"][:, 1] = ncY
        circles[path]["centers"][:, 2] = ncR
    return circles


def Task2c(datasets, circles, imageReader, params, display_results=True):
    """
    Estimate the AO's Height Above Ground.

    Use appropriate methods to estimate the vertical distance from
    the AO's lowest point to the ground plane(in meters).

    Args:
        datasets:
        circles:
        imageReader:
        params (Dict)
        display_results (bool)
        # distOfCams (float): The distance between 2 cameras.

    Returns:
        List[List[float]]: Depth of each image.
    """
    pairDatasets = dataset_to_image_pair(datasets, circles)
    depthList = []
    for pairDataset in pairDatasets:
        distOfCams = params[os.path.join(pairDataset["path"], "left")]["distOfCams"]
        leftImages = pairDataset["left"]
        rightImages = pairDataset["right"]
        depths = []
        for i in range(len(leftImages)):
            leftImagePath = leftImages[i]["path"]
            rightImagePath = rightImages[i]["path"]

            leftImage = imageReader.read_image_with_calibration(leftImagePath)
            rightImage = imageReader.read_image_with_calibration(rightImagePath)

            leftMask = pairDataset["left_circle"]["masks"][i]
            rightMask = pairDataset["right_circle"]["masks"][i]

            lCY = pairDataset["left_circle"]["centers"][i][1]
            rCY = pairDataset["right_circle"]["centers"][i][1]

            deltaY = (lCY - rCY)
            if deltaY < 0:
                deltaY = -deltaY
                leftImage = leftImage[:-deltaY, :, :]
                rightImage = rightImage[deltaY:, :, :]
                mask = leftMask[:-deltaY, :]
            elif deltaY > 0:
                rightImage = rightImage[:-deltaY, :, :]
                leftImage = leftImage[deltaY:, :, :]
                mask = leftMask[deltaY:, :]
            else:
                mask = leftMask

            stereo = cv2.StereoSGBM_create(
                minDisparity=1,
                numDisparities=64,
                blockSize=7,
                P1=8 * 3 * 7 ** 2,
                P2=32 * 3 * 7 ** 2,
                disp12MaxDiff=10,
                uniquenessRatio=2,
                speckleWindowSize=50,
                speckleRange=16
            )
            disparity = stereo.compute(leftImage, rightImage).astype(np.float32) / 16.0

            disp = disparity * mask
            f = imageReader.fy
            depth = f * distOfCams / (disp + 1)
            depth = depth * mask
            depths.append(np.mean(depth[depth != f * distOfCams]))

            if display_results:
                plt.title('Estimated Disparity')
                plt.imshow((disparity - disparity.min()) / (disparity.max() - disparity.min()), cmap='gray')
                plt.axis("off")
                plt.show()

                plt.title('Masked depth')
                plt.imshow(depth / np.max(depth), cmap='gray')
                plt.axis("off")
                plt.show()
        print(f"Average Depth: {np.mean(depths)}")
        depthList.append(depths)
    return depthList
