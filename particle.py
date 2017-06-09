"""
 generate vocabulary,size is 30,save as voc.mat
"""

import cv2
import numpy as np
from scipy import io
import sys


train_video=sys.argv[1]
output_voc=sys.argv[2]

cap = cv2.VideoCapture(train_video)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)


detector = cv2.xfeatures2d.SIFT_create()  # create a feature detector


bo = cv2.BOWKMeansTrainer(30) #number of voc is 30


while (ret):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('gen_voc', rgb)
        k = cv2.waitKey(30) & 0xff
        prvs = next

        kp1, desc1 = detector.detectAndCompute(rgb, None)  # find feature

        try:
            bo.add(desc1)
            print (desc1.shape[0])
        except:
            print('exception occur')

voc = bo.cluster()
io.savemat(output_voc, {output_voc[0:-4]: voc})
cap.release()
cv2.destroyAllWindows()
