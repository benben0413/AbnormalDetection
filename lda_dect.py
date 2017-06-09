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
    return ave,arr



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
lh = np.zeros(test.shape[0]/10)
j=0
for i in range(test.shape[0]):
    print(i)
    ll[i] = lda.score(test[i, :].reshape(1, -1))
for i in range(0, test.shape[0], 10):
    if (i + 10 >= test.shape[0]):
        break
    else:
        ave,arr=calAverage(ll, i, i + 10)
        lh[j]=ave
        j=j+1
io.savemat(output_likelihood, {output_likelihood[0:-4]: lh})
print("end")
