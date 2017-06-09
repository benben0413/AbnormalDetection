"""
 generate vocabulary,size is 30,save as voc.mat
"""

import cv2
import numpy as np
from scipy import io
import sys

def draw_flow(img, flow, step=2):
    h, w = img.shape[:2]
    win_h = 3
    win_w = 3
    y, x = np.mgrid[win_h:h:step, win_w:w:step].reshape(2, -1).astype(int)

    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, 0, (0, 0, 255))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis




train_video=sys.argv[1]
output_voc=sys.argv[2]

cap = cv2.VideoCapture(train_video)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)


detector = cv2.xfeatures2d.SURF_create()  # create a feature detector


bo = cv2.BOWKMeansTrainer(30) #number of voc is 30


while (ret):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 0] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('gen_voc', rgb)
        k = cv2.waitKey(30) & 0xff
        prvs = next

        kp1, desc1 = detector.detectAndCompute(rgb, None)  # find feature

        try:
            bo.add(desc1)

        except:
            print('exception occur')

voc = bo.cluster()
io.savemat(output_voc, {output_voc[0:-4]: voc})
cap.release()
cv2.destroyAllWindows()
