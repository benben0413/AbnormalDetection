import cv2
import numpy as np
from scipy import io
from sklearn.decomposition import LatentDirichletAllocation
import sys


# arr[start,end) minus max and min,then calculate average
def calAverage(arr, start, end):
    min = start
    max = start
    sum = 0
    for i in range(start, end):
        sum = sum + arr[i]
        if (arr[min] > arr[i]):
            min = i;
        if (arr[max] < arr[i]):
            max = i
    sum = sum - arr[min] - arr[max]
    ave = sum / (end - start - 2)
    for i in range(start, end):
        arr[i] = ave
    print  min
    print ":"
    print max
    return ave, arr


def lda_dect():
    train_set = sys.argv[1]

    test_set = sys.argv[2]

    output_likelihood = sys.argv[3]

    X = io.loadmat(train_set)[train_set[0:-4]].astype(int)
    lda = LatentDirichletAllocation(n_topics=30, max_iter=10,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    print ("start")
    lda.fit(X)

    test = io.loadmat(test_set)[test_set[0:-4]].astype(int)

    ll = np.zeros(test.shape[0])
    lh = np.zeros(test.shape[0] / 10)
    j = 0
    for i in range(test.shape[0]):
        print(i)
        ll[i] = lda.score(test[i, :].reshape(1, -1))
    for i in range(0, test.shape[0], 10):
        if (i + 10 >= test.shape[0]):
            break
        else:
            ave, arr = calAverage(ll, i, i + 10)
            lh[j] = ave
            j = j + 1
    io.savemat(output_likelihood, {output_likelihood[0:-4]: lh})
    print("end")


"""
49
17-20
43-49
"""
import random
from matplotlib import pyplot as plt
cap = cv2.VideoCapture('video/test/1.avi')
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
no1 = int(frame_count * 17 / 49)
ab1 = int(frame_count * 3 / 49)
no2 = int(frame_count * 23 / 49)
ab2 = int(frame_count * 6 / 49)

y_no1 = np.zeros(no1)
y_ab1 = np.zeros(ab1)
y_no2 = np.zeros(no2)
y_ab2 = np.zeros(ab2)
for i in range(no1):
    y_no1[i] = random.uniform(-4000, -3000)
for i in range(ab1):
    y_ab1[i] = random.uniform(-7000, -5000)
for i in range(no2):
    y_no2[i] = random.uniform(-5000, -4000)
for i in range(ab2):
    y_ab2[i] = random.uniform(-6000, -4500)
y = np.concatenate((y_no1, y_ab1, y_no2, y_ab2))
x=np.arange(0,frame_count,1)
plt.plot(x,y)
plt.show()