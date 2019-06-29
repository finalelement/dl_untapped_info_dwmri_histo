import os

def split_X_8th_order(X):

    x1 = X[:, 0]
    x2 = X[:, 1:6]
    x3 = X[:, 6:15]
    x4 = X[:, 15:28]
    x5 = X[:, 28:]

    return x1,x2,x3,x4,x5