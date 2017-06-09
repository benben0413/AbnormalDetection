"""
input a video,generate doc_voc matrix by the voc
"""

import cv2
import numpy as np
from scipy import io
import sys

video = sys.argv[1]
voc = sys.argv[2]
output_doc=sys.argv[3]
voc = io.loadmat(voc)[voc[0:-4]]


cap = cv2.VideoCapture(video)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)

detector = cv2.xfeatures2d.SIFT_create()  # create a feature detector
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
bowDE = cv2.BOWImgDescriptorExtractor(detector, matcher)
bowDE.setVocabulary(voc)

doc = np.ones(30)
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

        cv2.imshow('gen_doc', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalfb.png', frame2)
            cv2.imwrite('opticalhsv.png', rgb)
        prvs = next

        try:
            kp1, desc1 = detector.detectAndCompute(rgb, None)
            bdes = bowDE.compute(rgb, kp1) * desc1.shape[0]
            bdes = bdes.astype(int)
            doc = np.vstack((doc, bdes))
            #print (bdes)

        except BaseException, e:
            print('exception occur', e)

cap.release()
io.savemat(output_doc, {output_doc[0:-4]: doc})
cv2.destroyAllWindows()
