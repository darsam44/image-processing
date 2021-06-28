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

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 311594964


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """

    rgb_weights = [0.2989, 0.5870, 0.1140]
    Image = cv2.imread(filename)
    convertImg = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)

    if representation == LOAD_GRAY_SCALE:
        convertImg = np.dot(convertImg[...,:3], rgb_weights)

    #Normalize
    normalizedImage = (convertImg - convertImg.min()) / (convertImg.max() - convertImg.min())

    return normalizedImage


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    Image = imReadAndConvert(filename, representation)

    if representation == LOAD_RGB:
        plt.imshow(Image)

    if representation == LOAD_GRAY_SCALE:
        plt.imshow(Image, cmap='gray')

    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """

    yiqScaler = np.array([[0.299,  0.587,  0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523,  0.311]])

    shape = imgRGB.shape
    img = np.dot(imgRGB.reshape(-1, 3), yiqScaler.transpose())
    img = img.reshape(shape)
    return img


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    yiqScaler = np.array([[0.299,  0.587,  0.114],
                          [0.596, -0.275, -0.321],
                          [0.212, -0.523,  0.311]])

    shape = imgYIQ.shape
    img = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiqScaler).transpose())
    img = img.reshape(shape)
    return img


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    img = np.floor(255 * imgOrig).astype(np.uint8)

    originHisto = np.bincount(img.flatten(), minlength=256)

    pixelsum = np.sum(originHisto)  # sum all the pixels in the matrix

    Vector = np.floor(255 * np.cumsum(originHisto / pixelsum)).astype(np.uint8)
    flat_img = list(img.flatten())
    eq_img_list = [Vector[i] for i in flat_img]

    equalizedImage = np.reshape(np.asarray(eq_img_list), imgOrig.shape)

    returnHistogram = np.bincount(equalizedImage.flatten(), minlength=256)

    return equalizedImage, originHisto, returnHistogram


# chekc if Image is RGB or GRAY
def CheckRGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    IsRGB = False
    # if the image is RGB the bool will be true
    if len(imgOrig.shape) == 3:
        IsRGB = True

    if (IsRGB):
        YIQImg = transformRGB2YIQ(imgOrig)
        imgOrig = np.copy(YIQImg[:, :, 0])
        return IsRGB, imgOrig, YIQImg
    else:
        return IsRGB, np.copy(imgOrig), None
    pass

def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):

    IsRGB, Image, YIQImg = CheckRGB(imOrig)
    img = Image * 255
    hist, temp = np.histogram(img.flatten(), 256, [0, 255])

    z_arr = np.arange(0, 256, int(255 / nQuant))
    z_arr[nQuant] = 255

    Iamge_list = list()
    mse_list = list()

    for k in range(0, nIter):
        q_arr = [np.average(np.arange(z_arr[k], z_arr[k + 1] + 1), weights=hist[z_arr[k]: z_arr[k + 1] + 1]) for k in range(len(z_arr) - 1)]
        q_arr = np.round(q_arr).astype(int)

        New_Img = img.copy()

        if IsRGB:
            for i in range(1, nQuant + 1):
                New_Img[(New_Img > z_arr[i - 1]) & (New_Img < z_arr[i])] = q_arr[i - 1]
        else:
            for i in range(1, nQuant + 1):
                New_Img[(New_Img >= z_arr[i - 1]) & (New_Img < z_arr[i])] = q_arr[i - 1]

        # Error
        MSE = pow(np.power(img - New_Img, 2).sum(), 0.5)/img.size
        mse_list.append(MSE)

        if IsRGB:
            YIQImg[:, :, 0] = New_Img / (New_Img.max() - New_Img.min())
            New_Img = transformYIQ2RGB(YIQImg)

        # move boundaries
        for bound in range(1, len(z_arr)-1):
            z_arr[bound] = np.round((q_arr[bound - 1] + q_arr[bound]) / 2)

        Iamge_list.append(New_Img)

    return Iamge_list, mse_list
