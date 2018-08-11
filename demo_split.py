# -*- coding: utf-8 -*-
from sky.cv import *


def demo_split(filename):

    def create_trackbar(winname):
        cv2.createTrackbar('pos', winname, 130, 255, lambda x: x)
        cv2.createTrackbar('gauss', winname, 1, 1, lambda x: x)
        cv2.createTrackbar('ksize', winname, 3, 31, lambda x: x)
        cv2.createTrackbar('cnt', winname, 0, 140, lambda x: x)

    def get_trackbar_pos(winname):
        pos = cv2.getTrackbarPos('pos', winname)
        gauss = cv2.getTrackbarPos('gauss', winname)
        ksize = cv2.getTrackbarPos('ksize', winname)
        index = cv2.getTrackbarPos('cnt', winname)
        return pos, gauss, ksize, index

    def is_parent(hi):
        return hi[0][index][0] == -1 and \
               hi[0][index][1] == -1 and \
               hi[0][index][3] == -1 and \
               hi[0][index][2] != -1

    img = CVTest(filename)
    img.resize(None, 0.12, 0.12)
    mainwin = 'thr'
    cv2.namedWindow(mainwin)
    create_trackbar(mainwin)

    parent_index = None
    while True:
        # 二值化
        pos, gauss, ksize, index = get_trackbar_pos(mainwin)
        gray = img.gray
        if gauss:
            gray = cv2.GaussianBlur(img.gray, (ksize, ksize), 0)
        ret, thr = cv2.threshold(gray, pos, 255, cv2.THRESH_BINARY)
        # 显示二值图
        cv2.imshow(mainwin, thr)

        # 轮廓
        what, cnts, hi = cv2.findContours(thr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 复制一份原图,设置边界矩形
        im = img.im.copy()
        out_side = 10
        # 判断
        if index >= len(cnts):
            cv2.drawContours(im, cnts, -1, (0, 0, 255), 2)
        else:
            cv2.drawContours(im, cnts, index, (0, 0, 255), 2)
            # 边界矩形
            x, y, w, h = cv2.boundingRect(cnts[index])
            im = cv2.rectangle(im, (x-out_side, y-out_side), (x+w+out_side, y+h+out_side), (0, 255, 0), 2)

            if is_parent(hi):
                parent_index = index
                print('边缘')
            else:
                if parent_index is not None and hi[0][index][3] == parent_index:
                    print('数字')
            print(hi[0][index])

        # print(len(cnts))
        cv2.imshow('im', im)

        if cv2.waitKey(100) == ord('q'):
            break


if __name__ == '__main__':
    demo_split('src/20180805160122.jpg')
