import numpy as np
from matplotlib import pyplot as plt

x1=np.array([0,0.022,0.0710,0.155,0.331,0.411,1])
y1=np.array([0,0.351,0.451,0.752,0.842,1,1])
p1,=plt.plot(x1,y1,'r')

x2=np.array([0,0.034,0.073,0.189,0.351,0.42,1])
y2=np.array([0,0.162,0.352,0.568,0.711,1,1])
p2,=plt.plot(x2,y2,'b')

plt.legend([p1,p2],['force flow','optical flow'])

plt.ylabel('True Positive')
plt.xlabel('False Positive')


plt.show()
