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

import cv2

from ex1_utils import LOAD_GRAY_SCALE
from ex1_utils import LOAD_RGB

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """

    img = cv2.imread(img_path)
    if rep == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def tool_bar(val):
        gamma = (0.02 + val / 100.0)
        new_img = pow(img/255, gamma)
        cv2.imshow('Gamma bar', new_img)

    cv2.namedWindow('Gamma bar')
    cv2.createTrackbar('Gamma', 'Gamma bar', 100, 200, tool_bar)
    cv2.imshow('Gamma bar', img)
    cv2.waitKey(0)

def main():
     gammaDisplay('testImg1.jpg',LOAD_RGB)


if __name__ == '__main__':
    main()

