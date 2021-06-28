from ex2_utils import *
import matplotlib.pyplot as plt
import time
import cv2

beach = cv2.imread("beach.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
boxman = cv2.imread("boxman.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
coins = cv2.imread("coins.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
codeMonkey = cv2.imread("codeMonkey.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
pool_balls = cv2.imread("pool_balls.jpeg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
igoana = cv2.imread("igoana.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
sheep = cv2.imread("sheep.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
cow = cv2.imread("cow.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
lama = cv2.imread("lama.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)

def main():
    print(myID());

    conv1Demo()
    conv2Demo()
    derivDemo()
    blurDemo()
    edgeDemo()
    houghDemo()


def conv1Demo():
    a = np.array([1,2,3,4,5,4,3,2,1])
    b = np.array([0, 1, 0.5])
    print(np.sum(np.convolve(a, b) - conv1D(a, b)))


def conv2Demo():
    kernel = np.array([[0], [1], [0.5]])
    mypic = conv2D(igoana, kernel)
    filterpic = cv2.filter2D(igoana.astype(np.float32), -1, kernel.astype(np.float32), borderType=cv2.BORDER_REPLICATE).astype(
        np.float32)
    sub, img = plt.subplots(1, 2)
    sub.suptitle('conv2d', fontsize=16)
    img[0].imshow(mypic, cmap="gray")
    img[0].set_title('my pic')
    img[1].imshow(filterpic, cmap="gray")
    img[1].set_title('cv pic')
    plt.show()

def  derivDemo():
    directions, magnitude, x_der, y_der = convDerivative(cow)
    file, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    file.suptitle('Derivative', fontsize=16)
    ax1.imshow(directions, cmap="gray")
    ax1.set_title('directions')
    ax2.imshow(magnitude, cmap="gray")
    ax2.set_title('magnitude')
    ax3.imshow(x_der, cmap="gray")
    ax3.set_title('x_der')
    ax4.imshow(y_der, cmap="gray")
    ax4.set_title('y_der')
    plt.show()

def blurDemo():
    size = 25
    ans1 = blurImage1(sheep, size)
    ans2 = blurImage2(sheep, size)
    file, ax = plt.subplots(1, 2)
    file.suptitle('blur', fontsize=16)
    plt.title("blur")
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('my pic')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('cv pic')
    plt.show()

def edgeDemo():
    Sobel()
    ZeroCrossingSimple()
    Canny()


def Sobel():
    ans1, ans2 = edgeDetectionSobel(boxman)
    file, ax = plt.subplots(1, 2)
    file.suptitle('sobel', fontsize=16)
    ax[0].imshow(ans1, cmap="gray")
    ax[0].set_title('cv pic')
    ax[1].imshow(ans2, cmap="gray")
    ax[1].set_title('my pic')
    plt.show()

def ZeroCrossingSimple():
    ans1 = edgeDetectionZeroCrossingSimple(boxman)
    plt.gray()
    plt.title('Zero Crossing')
    plt.imshow(ans1)
    plt.show()


def Canny():
    cv_pic, my_pic = edgeDetectionCanny(lama, 100, 50)
    file, ax = plt.subplots(1, 2)
    file.suptitle('canny', fontsize=16)
    ax[0].imshow(cv_pic, cmap="gray")
    ax[0].set_title('cv pic')
    ax[1].imshow(my_pic, cmap="gray")
    ax[1].set_title('my pic')
    plt.show()

def houghDemo():
    list = houghCircle(coins, 60, 80)
    file, ax = plt.subplots()
    file.suptitle('hough', fontsize=16)
    ax.imshow(coins, cmap="gray")
    for c in list:
        circles = plt.Circle((c[0], c[1]), c[2], color='b', fill=False)
        ax.add_artist(circles)
    plt.show()

if __name__ == '__main__':
    main()

