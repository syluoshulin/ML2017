#!/usr/bin/python3.4
#-*-coding:UTF-8-*-

import sys
import csv
import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues

# Train a linear SVR

npzfile = np.load('large_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=800)
svr.fit(X, y)

# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
testdata = np.load(sys.argv[1])
test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)

print(svr.score(X,y))


with open(sys.argv[2], 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print(f'{i},{np.log(d)}', file=f)