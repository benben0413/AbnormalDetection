import cv2
import numpy as np
from scipy import io
from sklearn.decomposition import NMF, LatentDirichletAllocation

X=io.loadmat('b.mat')['matrix'].astype(int)
lda = LatentDirichletAllocation(n_topics=30, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
print ("start")
lda.fit(X)

test=io.loadmat('b.mat')['matrix'].astype(int)

ll=np.zeros(test.shape[0])



for i in range(test.shape[0]):
    print(i)
    ll[i]=lda.score(test[i,:].reshape(1,-1))
io.savemat('f.mat',{'matrix':ll})
print("end")


