import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy.ndimage import *
import cv2

def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 311594964


def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """
    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

    HIGH, WITHD = img_r.shape
    disparity_map = np.zeros((HIGH, WITHD, disp_range[1]))

    # uniform_filter does  avrege of k_size * k_size
    avrg_l = np.zeros((HIGH, WITHD))
    avrg_r = np.zeros((HIGH, WITHD))
    filters.uniform_filter(img_l, k_size, avrg_l)
    filters.uniform_filter(img_r, k_size, avrg_r)

    # normalized image
    normaliz_l = img_l - avrg_l
    normaliz_r = img_r - avrg_r

    for shift_amount in range(disp_range[1]):

        shifted_to_right = np.roll(normaliz_r, shift_amount)

        filters.uniform_filter(normaliz_l * shifted_to_right, k_size, disparity_map[:, :, shift_amount])
        disparity_map[:, :, shift_amount] = disparity_map[:, :, shift_amount] ** 2

    # choose the best at second place
    return np.argmax(disparity_map, axis=2)


def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    HIGH, WITHD = img_r.shape
    disp_map = np.zeros((HIGH, WITHD, disp_range[1]))

    # uniform_filter does  avrege of k_size * k_size
    mean_l = np.zeros((HIGH, WITHD))
    mean_r = np.zeros((HIGH, WITHD))
    filters.uniform_filter(img_l, k_size, mean_l)
    filters.uniform_filter(img_r, k_size, mean_r)

    # normalized image
    normaliz_l = img_l - mean_l
    normaliz_r = img_r - mean_r

    s = np.zeros((HIGH, WITHD))
    s_l = np.zeros((HIGH, WITHD))
    s_r = np.zeros((HIGH, WITHD))

    filters.uniform_filter(normaliz_l * normaliz_l, k_size, s_l)

    for shift_amount in range(disp_range[1]):

        shifted_to_right = np.roll(normaliz_r, shift_amount - disp_range[0])
        filters.uniform_filter(normaliz_l * shifted_to_right, k_size, s)
        filters.uniform_filter(shifted_to_right * shifted_to_right, k_size, s_r)

        # Save ncc score
        disp_map[:, :, shift_amount] = s / np.sqrt(s_r * s_l)

    # Choose the best depth for each pixel
    return np.argmax(disp_map, axis=2)


def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    A = np.zeros((2 * len(src_pnt), 9))
    for i in range(len(src_pnt)):
        A[i * 2:i * 2 + 2] = np.array([[-src_pnt[i][0], -src_pnt[i][1], -1, 0, 0, 0, src_pnt[i][0] * dst_pnt[i][0],
                                        src_pnt[i][1] * dst_pnt[i][0], dst_pnt[i][0]],
                                       [0, 0, 0, -src_pnt[i][0], -src_pnt[i][1], -1, src_pnt[i][0] * dst_pnt[i][1],
                                        src_pnt[i][1] * dst_pnt[i][1], dst_pnt[i][1]]])

    # svd on mat
    a, b, c_transposed = np.linalg.svd(A, full_matrices=True)
    C = np.transpose(c_transposed)

    res = C[:, -1].reshape(3, 3)
    res /= C[:, -1][-1]

    error = 0
    for i in range(len(src_pnt)):
        x = src_pnt[i, 0]
        y = src_pnt[i, 1]
        Ah = np.array([x,y,1])
        Ah = res.dot(Ah)
        Ah /= Ah[2]
        Ah = Ah[0:-1]
        error += np.sqrt(sum(Ah - dst_pnt[i])**2)

    return res, error



def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()
    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, 'r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()

    # display  first image
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    src_p = []
    fig2 = plt.figure()


    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, 'r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display second image
    cid2 = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)

    # My code
    HOMO , e = computeHomography(src_p, dst_p)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            Ah = np.array([j, i, 1])
            Ah = HOMO.dot(Ah)
            Ah /= Ah[2]
            dst_img[int(Ah[1]), int(Ah[0])] = src_img[i,j]

    plt.imshow(dst_img)
    plt.show()
