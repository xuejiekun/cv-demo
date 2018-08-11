# -*- coding: utf-8 -*-
import cv2
from func import  *


def demo_contour(filename):
    set_click('im')
    im = cv2.imread(filename)

    # 灰度图
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    # 轮廓
    img, cnts, hi = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 轮廓控制控件
    cv2.createTrackbar('cnt', 'im', 0, len(cnts), lambda x: x)

    while True:
        index = cv2.getTrackbarPos('cnt', 'im')


        if index >= len(cnts):  # 全部绘制
            cnt_img = cv2.drawContours(im.copy(), cnts, -1, (255, 255, 0), 2)
            print('contours:{}'.format(len(cnts)))

        else:   # 部分绘制
            cnt_img = cv2.drawContours(im.copy(), cnts, index, (255, 255, 0), 2)

            # 矩
            m = cv2.moments(cnts[index])

            # 面积,周长
            area = cv2.contourArea(cnts[index])
            length = cv2.arcLength(cnts[index], True)
            print('area:{} length:{}'.format(area, length))

            # 重心
            x = int(m['m10'] / m['m00'])
            y = int(m['m01'] / m['m00'])
            cnt_img = cv2.circle(cnt_img, (x, y), 5, (0, 0, 255), -1)

            # 边界矩形
            x, y, w, h = cv2.boundingRect(cnts[index])
            cnt_img = cv2.rectangle(cnt_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow('im', cnt_img)
        if cv2.waitKey(100) == ord('q'):
            break


if __name__ == '__main__':
    demo_contour('src/cv2_test.jpg')
