import copy
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_largest_circle(image, min_dist=500, param1=100, param2=20, min_radius=0, max_radius=0,
                          display_results=True):
    """
    Detects the largest circle in an image and returns its radius.

    Args:
        image (str): The image file.
        min_dist (int): Minimum distance between the centers of detected circles.
        param1 (int): First method-specific parameter for cv2.HoughCircles.
        param2 (int): Second method-specific parameter for cv2.HoughCircles.
        min_radius (int): Minimum circle radius.
        max_radius (int): Maximum circle radius.
        display_results (bool): Whether to display the detected circle and binary mask.

    Returns:
        int: The radius of the largest circle if detected, otherwise None.
    """
    image = copy.copy(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.2,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        circles = np.int16(np.around(circles))
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        mask = np.zeros_like(gray)
        cv2.circle(mask, (largest_circle[0], largest_circle[1]), largest_circle[2], 255, thickness=-1)
        mask[mask == 255] = 1

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

        return largest_circle
    else:
        print("No circles detected.")
        return None


def gaussian_kernel(size=5, sigma=1):
    """生成高斯核函数"""
    kernel = np.linspace(-(size // 2), size // 2, size)
    kernel = np.exp(-0.5 * (kernel / sigma) ** 2)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


def get_match_pts(imgDict1, imgDict2, method="sift", display_results=True):
    """
    Get match points between 2 images with a mask
    to control the matching range

    Args:
        imgDict1 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2 (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        method (str): "sift" or "ord"
        display_results (bool)

    Returns:
        ndarray: Key point pairs in the first image. [N, 2, 2]
    """
    img1 = imgDict1["image"]
    img2 = imgDict2["image"]

    if method == "sift":
        sift = cv2.SIFT_create()
        kpts1, descriptors1 = sift.detectAndCompute(img1, imgDict1["mask"])
        kpts2, descriptors2 = sift.detectAndCompute(img2, imgDict2["mask"])
    else:
        orb = cv2.ORB_create()
        kpts1, descriptors1 = orb.detectAndCompute(img1, imgDict1["mask"])
        kpts2, descriptors2 = orb.detectAndCompute(img2, imgDict2["mask"])

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_img = cv2.drawMatches(
        img1, kpts1, img2, kpts2, matches[:100], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    if display_results:
        scale_percent = 50
        new_width = int(matched_img.shape[1] * scale_percent / 100)
        new_height = int(matched_img.shape[0] * scale_percent / 100)
        dim = (new_width, new_height)

        resized_img = cv2.resize(matched_img, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow("Keypoint Matches", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kptsPairs = []
    for m in matches:
        kpt1 = kpts1[m.queryIdx].pt
        kpt2 = kpts2[m.trainIdx].pt
        kptsPairs.append([kpt1, kpt2])

    kptsPairs = np.array(kptsPairs)
    return kptsPairs


def detect_outliers(data, threshold=3):
    """
    Detect the outliers and filter the data.

    Args:
        data (ndarray)
        threshold (int)

    Returns:
        ndarray: filtered data
        ndarray: outliers
    """
    mean = np.mean(data)
    std_dev = np.std(data)

    outliers = []
    filteredData = []
    for value in data:
        if abs(value - mean) > threshold * std_dev:
            outliers.append(value)
        else:
            filteredData.append(value)
    return np.array(filteredData), np.array(outliers)


def cal_rotate_degree(imgDict1, imgDict2, kptsPairs):
    """
    Calculate the rotate degree of matched points between 2 images.
    
    Args:
        imgDict1: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        kptsPairs (ndarray): [N, 2, 2]

    Returns:
        ndarray: The rotate degree of matched points without outliers.
    """
    center1 = imgDict1["center"]
    center2 = imgDict2["center"]
    kpts1 = kptsPairs[:, 0, :]
    kpts2 = kptsPairs[:, 1, :]
    kvecs1 = np.array([kpts1[:, 0] - center1[0], kpts1[:, 1] - center1[1]]).T
    kvecs2 = np.array([kpts2[:, 0] - center2[0], kpts2[:, 1] - center2[1]])

    cos = np.dot(kvecs1, kvecs2) / (np.linalg.norm(kvecs1) * np.linalg.norm(kvecs2))
    cos = np.diagonal(cos)
    sin = np.cross(kvecs1, kvecs2.T) / (np.linalg.norm(kvecs1) * np.linalg.norm(kvecs2))
    theta = np.arctan2(sin, cos)

    theta, _ = detect_outliers(theta)
    return theta


def cal_theta_any_view(imgDict1, imgDict2, kptsPairs, elevation=0):
    """
    Calculate the rotation degree of matched points between 2 images
    from any view.

    Args:
        imgDict1: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        kptsPairs (ndarray): [N, 2, 2]

    Returns:
        ndarray: The rotate degree of matched points without outliers.
    """
    center1 = imgDict1["center"]
    center2 = imgDict2["center"]
    r1 = center1[2] * np.cos(elevation)
    r2 = center2[2] * np.cos(elevation)
    bound1 = [center1[1] - 20, center1[1] + 20]
    bound2 = [center2[1] - 20, center2[1] + 20]

    kpts1 = kptsPairs[:, 0, :]
    kpts2 = kptsPairs[:, 1, :]

    boundMask1 = np.logical_and(bound1[0] < kpts1[:, 1], bound1[1] > kpts1[:, 1])
    boundMask2 = np.logical_and(bound2[0] < kpts2[:, 1], bound2[1] > kpts2[:, 1])
    boundMask = np.logical_and(boundMask1, boundMask2)
    kpts1 = kpts1[boundMask]
    kpts2 = kpts2[boundMask]

    # print("="*20)
    # print(center1)
    # print(center2)
    dist1 = kpts1[:, 0] - center1[0]
    dist2 = kpts2[:, 0] - center2[0]
    theta1 = np.arcsin(dist1 / r1)
    theta2 = np.arcsin(dist2 / r2)
    theta = theta2 - theta1
    # pair = np.array([dist1, dist2])
    # print(pair.T)
    # dist = dist2 - dist1
    # dist, _ = detect_outliers(dist)
    return theta


def cal_omega(imgDict1, imgDict2, method="sift", display_results=True):
    """
    Calculate the OA's rotation velocity by given two image.
    Args:
        imgDict1: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        method (str): "sift" or "ord"
        display_results (bool)

    Returns:
        float
    """
    kptsPairs = get_match_pts(imgDict1, imgDict2, method, display_results)
    theta = cal_rotate_degree(imgDict1, imgDict2, kptsPairs)
    if "ts" in imgDict1.keys():
        deltaTime = (imgDict2["ts"] - imgDict1["ts"]) / 1e7
    else:
        deltaTime = 1. / imgDict1["fps"]

    # print(theta)
    omega = np.mean(theta) / deltaTime
    return omega


def cal_omega_any_view(imgDict1, imgDict2, method="sift", display_results=True):
    """
    Calculate the OA's rotation velocity by given two image from any view.
    Args:
        imgDict1: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the first image.
        imgDict2: (Dict{
                "image": ndarray,
                "center": ndarray,
                "mask": ndarray,
            }): Information about the second image.
        method (str): "sift" or "ord"
        display_results (bool)

    Returns:
        float
    """
    kptsPairs = get_match_pts(imgDict1, imgDict2, method, display_results)
    theta = cal_theta_any_view(imgDict1, imgDict2, kptsPairs)

    if "ts" in imgDict1.keys():
        deltaTime = (imgDict2["ts"] - imgDict1["ts"]) / 1e7
    else:
        deltaTime = 1. / imgDict1["fps"]

    # print(deltaTime)
    # vel = np.mean(s) / deltaTime
    # # print(vel)
    # return vel
    pNum = len(theta[theta > 0])
    nNum = len(theta[theta < 0])

    if pNum > nNum:
        theta = theta[theta > 0]
    else:
        theta = [theta[theta < 0]]
    omega = np.mean(theta) / deltaTime
    return omega

