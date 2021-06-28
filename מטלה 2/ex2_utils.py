"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""

from typing import List

import matplotlib.pyplot as plt
import cv2
import numpy as np

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 311594964

def conv1D(inSignal:np.ndarray,kernel1:np.ndarray)->np.ndarray:
    """
       Convolve a 1-D array with a given kernel
       :param inSignal: 1-D array
       :param kernel1: 1-D array as a kernel
       :return: The convolved array
       """
    kernel = kernel1[::-1] #flip the array
    temp = np.pad(inSignal, (len(kernel) - 1, len(kernel) - 1), 'constant')
    result = np.zeros(len(temp) - len(kernel) + 1)
    for i in range(0, len(temp) - len(kernel) + 1):
        result[i] = np.multiply(temp[i:i + len(kernel)], kernel).sum()

    return result

def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Convolve a 2-D array with a given kernel
    :param inImage: 2D image
    :param kernel2: A kernel
    :return: The convolved image
    """

    kernel = kernel2

    kernel = kShape(kernel) # take the kernel and reshape him to square

    temp = np.pad(inImage, (kernel.shape[0] // 2, kernel.shape[1] // 2), 'edge').astype('float32')

    result = np.ndarray(inImage.shape).astype('float32')

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = (temp[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel).sum()

    return result

def kShape(k: np.ndarray) -> np.ndarray:

    if (len(k.shape) == 1):
        k = np.pad(k.reshape(1, len(k)).transpose(), (len(k) // 2, len(k) // 2), 'constant')
        k = k[1:k.shape[0] - 1, :]

    if (k.shape[1] == 1):
        k = np.pad(k.reshape(1, len(k)).transpose(), (len(k) // 2, len(k) // 2), 'constant')
        k = k[1:k.shape[0] - 1, :]

    return k

def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calculate gradient of an image
    :param inImage: Grayscale iamge
    :return: (directions, magnitude,x_der,y_der)
    """
    # the kernel for the derivative
    kernel = np.array([ [0, 1, 0],
                        [0, 0, 0],
                        [0, -1, 0]])

    xDer = conv2D(inImage, kernel.transpose())
    yDer = conv2D(inImage, kernel)
    magG = np.sqrt(np.power(xDer, 2) + np.power(yDer, 2))
    direction = np.arctan2(yDer, xDer)
    return direction, magG, xDer, yDer


def gaussian_kernel(kernel_size, sigma=1) -> np.ndarray:
  kernel_size = int(kernel_size) // 2
  x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
  normal = 1 / (2.0 * np.pi * sigma**2)
  g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
  return g

def blurImage1(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel
    :param inImage: Input image
    :param kernelSize: Kernel size
    :return: The Blurred image
    """
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    kernel = gaussian_kernel(kernel_size, sigma)
    blur = conv2D(in_image, kernel)
    return blur

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

def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    """
    Detects edges using the Sobel method
    :param img: Input image
    :param thresh: The minimum threshold for the edge response
    :return: opencv solution, my implementation
    """

    s = np.array([[1, 0, -1],
                  [2, 0, -2],
                  [1, 0, -1]])

    thresh *= 255
    mcon = np.sqrt((conv2D(img, s) ** 2 + conv2D(img, s.transpose()) ** 2))

    mResualt = np.ndarray(mcon.shape)
    mResualt[mcon > thresh] = 1
    mResualt[mcon < thresh] = 0

    cCon = cv2.magnitude(cv2.Sobel(img, -1, 1, 0), cv2.Sobel(img, -1, 0, 1))
    cResualt = np.ndarray(cCon.shape)
    cResualt[cCon > thresh] = 1
    cResualt[cCon < thresh] = 0

    return cResualt, mResualt


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    """
    Detecting edges using the "ZeroCrossing" method
    :param I: Input image
    :return: Edge matrix
    """
    ker = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]])

    conv = conv2D(img , ker)
    result = np.zeros(conv.shape)

    # Check for a zero crossing around (x,y)
    for i in range(0, conv.shape[0]):
        for j in range(0, conv.shape[1]):
            try:
                if conv[i, j] == 0:
                    if (conv[i, j + 1] > 0 and conv[i, j - 1] < 0) or (conv[i, j + 1] < 0 and conv[i, j - 1] > 0) or (
                            conv[i + 1, j] > 0 and conv[i - 1, j] < 0) or (conv[i + 1, j] < 0 and conv[i - 1, j] > 0):
                        result[i, j] = 1
                elif conv[i, j] > 0:
                    if conv[i, j + 1] < 0 or conv[i + 1, j] < 0:
                        result[i, j] = 1
                else:
                    if conv[i, j + 1] > 0 or conv[i + 1, j] > 0:
                        result[i, j] = 1

            except IndexError as e:
                pass
    return result



def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    """
    Detecting edges usint "Canny Edge" method
    :param img: Input image
    :param thrs_1: T1
    :param thrs_2: T2
    :return: opencv solution, my implementation
    """


    magne = np.sqrt(np.power(cv2.Sobel(img, -1, 0, 1), 2) + np.power(cv2.Sobel(img, -1, 1, 0), 2))
    d = np.arctan2(cv2.Sobel(img, -1, 0, 1), cv2.Sobel(img, -1, 1, 0))
    mCanny = non_max_suppression(magne, d)

    for i in range(0, mCanny.shape[0]):
        for j in range(0, mCanny.shape[1]):
            try:
                if mCanny[i][j] <= thrs_2:
                    mCanny[i][j] = 0
                elif thrs_2 < mCanny[i][j] < thrs_1:
                    neighbor = mCanny[i - 1:i + 2, j - 1: j + 2]
                    if neighbor.max() < thrs_1:
                        mCanny[i][j] = 150
                    else:
                        mCanny[i][j] = 255
                else:
                    mCanny[i][j] = 255
            except IndexError as e:
                pass


    for i in range(0, mCanny.shape[0]):
        for j in range(0, mCanny.shape[1]):
            try:
                if mCanny[i][j] == 150:
                    neighbor = mCanny[i - 1:i + 2, j - 1: j + 2]
                    if neighbor.max() < thrs_1:
                        mCanny[i][j] = 0
                    else:
                        mCanny[i][j] = 255
            except IndexError as e:
                pass

    cvcan = cv2.Canny(img.astype(np.uint8), thrs_1, thrs_2)
    return cvcan, mCanny


def non_max_suppression(img: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    Preforming a non maximum suppuration to a given img using it's direction matrix
    Will first change the radians to degrees and make all between 0-180
    "Quantisize" the image to 4 groups and will check the neighbors according
    The is to make sure we will get the edges with less noise around them
    """
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.float32)
    angle = np.rad2deg(D)
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                # Check who is greater amount my neighbors
                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z

def houghCircle(img:np.ndarray,min_radius:float,max_radius:float)->list:
    """
    Find Circles in an image using a Hough Transform algorithm extension
    :param I: Input image
    :param minRadius: Minimum circle radius
    :param maxRadius: Maximum circle radius
    :return: A list containing the detected circles,
    [(x,y,radius),(x,y,radius),...]
        """

    imgc = cv2.Canny(img.astype(np.uint8), 100, 50)

    div = np.arctan2(cv2.Sobel(img, -1, 0, 1), cv2.Sobel(img, -1, 1, 0))

    tresh = 20

    hough = np.zeros((imgc.shape[0], imgc.shape[1], max_radius - min_radius))
    list = []

    for r in range(hough.shape[2]):
        for x in range(0, imgc.shape[1]):
            for y in range(0, imgc.shape[0]):
                if imgc[y, x] != 0:
                    try:
                        a1 = x + (r + min_radius) * np.cos(div[y, x])
                        b1 = y + (r + min_radius) * np.sin(div[y, x])
                        a2 = x - (r + min_radius) * np.cos(div[y, x])
                        b2 = y - (r + min_radius) * np.sin(div[y, x])
                        hough[int(a1), int(b1), r] += 1
                        hough[int(a2), int(b2), r] += 1

                    except IndexError as e:
                        pass

    for r in range(hough.shape[2]):
        for x in range(0, img.shape[0]):
            for y in range(0, img.shape[1]):
                if hough[x, y, r] > tresh:
                    list.append((x, y, min_radius + r))

    return list