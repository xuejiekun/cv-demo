# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

__all__ = ['create_rgb_trackbar', 'get_rgb_trackbar_pos', 'set_click', 'draw_hist']


def create_rgb_trackbar(winname):
    def nothing(x): pass

    cv2.createTrackbar('R', winname, 0, 255, nothing)
    cv2.createTrackbar('G', winname, 0, 255, nothing)
    cv2.createTrackbar('B', winname, 0, 255, nothing)


def get_rgb_trackbar_pos(winname):
    r = cv2.getTrackbarPos('R', winname)
    g = cv2.getTrackbarPos('G', winname)
    b = cv2.getTrackbarPos('B', winname)
    print(r, g, b)
    return r, g, b


def set_click(winname):
    cv2.namedWindow(winname)

    def mouse(event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('click:{}, {}'.format(x, y))

    cv2.setMouseCallback(winname, mouse)


def draw_hist(filename):
    im = cv2.imread(filename)
    fig, axes = plt.subplots(2, 2)

    # 灰度直方图
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    axes[1, 0].hist(gray.flatten(), bins=256, density=1)

    # RGB直方图
    colors = ['b', 'g', 'r']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([im], [i], None, [256], [0, 256])
        axes[1, 1].plot(hist, c=color)

    axes[0, 0].imshow(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    axes[0, 1].imshow(im[:,:,::-1])

    plt.show()
