import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    kernel_arr = ([kernel_size,kernel_size])
    sigma = 0.3 * ((kernel_arr[0] - 1) * 0.5 - 1) + 0.8
    kernel = cv2.getGaussianKernel(kernel_arr[0], sigma)
    blur = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return blur


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    matT = im1 - im2
    maty = cv2.Sobel(im1, -1, 0, 1)
    matx = cv2.Sobel(im1, -1, 1, 0)


    res1 = []
    res2 = []
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):
            try:
                x_from = i - win_size // 2
                x_to = i + 1 + win_size // 2
                y_from =j - win_size // 2
                y_to = j + 1 + win_size // 2

                #creat 5X5 matrix
                windowMatx = matx[x_from: x_to, y_from: y_to]
                windowMaty = maty[x_from: x_to, y_from: y_to]
                windowMatT = matT[x_from: x_to, y_from: y_to]
                if windowMatx.size < win_size * win_size:
                    break
                X_Y = np.concatenate((windowMatx.reshape((win_size * win_size, 1)), windowMaty.reshape((win_size * win_size, 1))), axis=1)
                #reshape Ix and Iy and add it to A

                T = (windowMatT.reshape((win_size * win_size, 1)))
                lamda= np.linalg.eigvals(np.dot(X_Y.T, X_Y))

                lamda = np.sort(lamda)
                if (lamda[1] >= lamda[0] > 1 and (lamda[1] / lamda[0]) < 100):
                    v = np.dot(np.dot(np.linalg.inv(np.dot(X_Y.T, X_Y)), X_Y.T), T)
                    res1.append(np.array([j, i]))
                    res2.append(v)

            except IndexError as e:
                pass

    return np.array(res1), np.array(res2)

def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    res = []
    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]
    res.append(img)


    for i in range(1, levels):
        img = blurImage2(img, 5)
        img = img[::2, ::2]
        res.append(img)
    return res




def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """

    if (len(img.shape) == 2):
        out = np.zeros((2 * img.shape[0], 2 * img.shape[1]), dtype=img.dtype)
    else:
        out = np.zeros((2 * img.shape[0], 2 * img.shape[1], img.shape[2]), dtype=img.dtype)
    out[::2, ::2] = img
    res = cv2.filter2D(out, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    return res




def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """


    res = []
    sigma = 1.1
    k = cv2.getGaussianKernel(5, sigma)
    k = k * k.transpose() * 4

    h = pow(2, levels) * (img.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img.shape[1] // pow(2, levels))
    img = img[:h, :w]

    pyrmlist = gaussianPyr(img, levels)
    orig_img = img.copy()
    for i in range(1, levels):
        exp = gaussExpand(pyrmlist[i], k)
        res.append(orig_img - exp)
        orig_img = pyrmlist[i]

    res.append(pyrmlist[levels - 1])

    return res



def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """

    lap_pyr.reverse()

    temp = lap_pyr.pop(0)
    base_img = temp
    sigma = 1.1
    guassian = cv2.getGaussianKernel(5, sigma)
    guassian = guassian * guassian.transpose() * 4



    for lap_img in lap_pyr:
        ex_img = gaussExpand(base_img, guassian)
        base_img = ex_img + lap_img

    lap_pyr.insert(0, temp)
    lap_pyr.reverse()
    return base_img


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """

    sigma = 1.1
    guassian = cv2.getGaussianKernel(5, sigma)
    guassian = guassian * guassian.transpose() * 4

    h = pow(2, levels) * (img_1.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_1.shape[1] // pow(2, levels))
    img_1 = img_1[:h, :w]

    h = pow(2, levels) * (img_2.shape[0] // pow(2, levels))
    w = pow(2, levels) * (img_2.shape[1] // pow(2, levels))
    img_2 = img_2[:h, :w]

    h = pow(2, levels) * (mask.shape[0] // pow(2, levels))
    w = pow(2, levels) * (mask.shape[1] // pow(2, levels))
    mask = mask[:h, :w]

    list_mask = gaussianPyr(mask, levels)
    list_img_1 = laplaceianReduce(img_1, levels)
    list_img_2 = laplaceianReduce(img_2, levels)

    curr = list_img_1[levels - 1] * list_mask[levels - 1] + (1 - list_mask[levels - 1]) * list_img_2[levels - 1]

    for i in range(levels - 2, -1, -1):
        curr = gaussExpand(curr, guassian) + list_img_1[i] * list_mask[i] + (1 - list_mask[i]) * list_img_2[i]

    naive = img_1 * mask + (1 - mask) * img_2

    return naive, curr

    pass
