# -*- coding: utf-8 -*-
import cv2


def videocap(dev, output='out.avi', winname='cv2'):
    cap = cv2.VideoCapture(dev)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(output, fourcc, 20, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow(winname, frame)
        else:
            break

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    out.release()


def videoplay(filename, fps=10, winname='cv2'):
    cap = cv2.VideoCapture(filename)
    for i in range(10):
        print(cap.get(i))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imshow(winname, frame)
        else:
            break

        if cv2.waitKey(int(1000/fps)) == ord('q'):
            break
    cap.release()
