# -*- coding: utf-8 -*-
import numpy as np

from func import  *
from video import  *


def create_selector(r, g, b):
    hig = np.uint8([[[b, g, r]]])
    return cv2.cvtColor(hig, cv2.COLOR_BGR2HSV)[0][0]


# 通过色彩范围选择
def demo_range(filename):
    set_click('image')
    im = cv2.imread(filename)

    while True:
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

        # 范围选择
        hig = create_selector(252, 252 , 96)
        low = hig - np.array([10, 5, 25], dtype='uint8')
        print('hig:{} low:{}'.format(hig, low))

        # 创建mask
        mask = cv2.inRange(hsv, low, hig)
        res = cv2.bitwise_and(im, im, mask=mask)

        # cv2.imshow('image', cv2.resize(im[10:220, 540:680], (280,420)))
        # cv2.imshow('image', im[10:220, 540:680])
        cv2.imshow('image', im)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    demo_range('src/20180730.113601.02.19.200.png')
