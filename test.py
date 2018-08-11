# -*- coding:utf-8 -*-
import time
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

from sky.cv import *
from wea.ImgTool import *


def test_aps():
    # 被测函数
    def test():
        print('test:{}'.format(datetime.now()))

    sched = BackgroundScheduler()
    sched.add_job(test, 'cron', second='0-10/5')
    sched.start()

    try:
        # 其他任务是独立的线程执行
        while True:
            time.sleep(2)
            print('sleep!')
    except KeyboardInterrupt:
        sched.shutdown(wait=False)
        print('Exit The Job!')


def test_cv2():
    # 从文件创建视频
    # f = lambda x: int(x.split('\\')[-1].split('.')[0])
    # create_gif_with_cv2('out.avi', 'test', 10, key=f)

    # 从视频分解帧
    save_as_frame_with_cv2('out2.avi', 'test2')


def split_it(filename, target_dir):

    def is_parent(hi, index):
        return hi[0][index][0] == -1 and \
               hi[0][index][1] == -1 and \
               hi[0][index][3] == -1 and \
               hi[0][index][2] != -1

    img = CVTest(filename)
    img.resize(None, 0.15, 0.15)

    # 二值
    ksize = 3
    gray = cv2.GaussianBlur(img.gray, (ksize, ksize), 0)
    ret, thr = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # 轮廓
    what, cnts, hi = cv2.findContours(thr.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 找外框
    parent_index = None
    for index, cnt in enumerate(cnts):
        if is_parent(hi, index):
            parent_index = index
            break

    # 记录边界矩形
    roi_list = []
    out_side = 7
    for index, cnt in enumerate(cnts):
        if parent_index is not None and hi[0][index][3] == parent_index:
            # roi
            x, y, w, h = cv2.boundingRect(cnts[index])
            roi_list.append((y-out_side, y+h+out_side, x-out_side, x+w+out_side))

    # 从roi构建img
    os.makedirs(target_dir, exist_ok=True)
    roi_list.sort(key=lambda x: x[2])
    img_list = []
    ct = 1
    for i in roi_list:
        # [y:y+h, x:x+w]
        im = 255 - cv2.resize(thr[i[0]:i[1], i[2]:i[3]], (20,20))
        img_list.append(im.flatten())
        # 保存图片
        cv2.imwrite(os.path.join(target_dir, '{}.jpg'.format(ct)), im)
        ct += 1

    return np.array(img_list)


def get_train_data(filename, save=None):
    img = CVTest(filename)
    img_list = []
    for i in np.vsplit(img.gray, 50):
        for j in np.hsplit(i, 100):
            img_list.append(j.flatten())

    if save:
        np.save(save, img_list)

    return np.array(img_list)


def save_pic(img_list):
    for i in range(10):
        down_dir = os.path.join('dig', '{}'.format(i))
        if not os.path.exists(down_dir):
            os.makedirs(down_dir, exist_ok=True)

        ct = 1
        for im in img_list[i*500:(i+1)*500]:
            file_name = os.path.join(down_dir, '{}_{}.jpg'.format(i, ct))
            cv2.imwrite(file_name, im)
            ct += 1


def except_data(res, rel_data):
    ct = 0
    for i,r in enumerate(res):
        if r[0] == int(rel_data[i]):
            ct += 1
    return ct/len(rel_data)


if __name__ == '__main__':
    # test_aps()
    # test_cv2()

    # 训练数据
    # 5000,400
    # traindata = get_train_data(r'G:\opencv\sources\samples\python2\data\digits.png', 'data').astype(np.float32)
    traindata = np.load('src/data.npy').astype(np.float32)
    labels = np.zeros((5000,1)).astype(np.float32)
    for i in range(10):
        labels[i*500:(i+1)*500] = i

    # 手写数据
    img_list = split_it('src/20180805160122.jpg', 'res/hand').astype(np.float32)
    print(img_list.shape, traindata.shape, labels.shape)

    knn = cv2.ml.KNearest_create()
    knn.train(traindata, cv2.ml.ROW_SAMPLE, labels)
    ret, results, neighbours, dist = knn.findNearest(img_list, 5)

    print("result:  {}".format(np.uint8(results)))
    print('except: {}'.format(except_data(np.uint8(results), '243019867')))
    # print("neighbours:  {}".format(neighbours))
    # print("distance:  {}".format(dist))
    pass
