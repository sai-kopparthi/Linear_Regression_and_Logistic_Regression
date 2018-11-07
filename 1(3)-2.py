from sklearn import datasets
import numpy as np
from scipy import sparse
#question 1(1)
def MSE(wttrain, xtest, yttest):
    print(xtest.shape)
    print(wtrain.shape)
    pred = xtest @ wttrain
    error = pred - yttest
    sumsq = magnitude(error)
    ntest = yttest.shape[0]
    sums = sumsq/ntest
    print("Final MSE", sums)
    return sums

#delta f
def deltag(wt,y_array,x_array,n):
    y = np.matrix(y_array)
    yt = y.T
    a = sparse.csr_matrix(x_array).multiply(sparse.csr_matrix(yt))
    b = a - yt
    c = x_array.T @ b
    d = (3/n)*c + wt
    return d

#def magnitude

def magnitude(vector):
    return np.linalg.norm(vector)


#gradient descent linear regression
def grad(n,e,w,iter,y_array,x_array):
    wa = w
    r_0 = magnitude(np.array((deltag(wa,y_array,x_array,n))))
    for i in range(iter):
        g = np.array(deltag(wa,y_array,x_array,n))
        b = magnitude(g) <= ( e * r_0)
        if b:
            wa = wa - n*deltag(wa,y_array,x_array,n)
        else:
             break
    return wa



#Question 1(3)
filename1 = "/home/sai/PycharmProjects/hw1/E2006.train"
X1,Y1 = datasets.load_svmlight_file(filename1)
print(X1.shape)
print(Y1.shape)
filename2 = "/home/sai/PycharmProjects/hw1/E2006.test"
X2,Y2 = datasets.load_svmlight_file(filename2)
print(X2.shape)
c = X2.shape[1]
X1 = X1[:,:c]
X1 = sparse.csr_matrix(X1)
X2 = sparse.csr_matrix(X2)
wtrain = grad(0.2,0.01,np.random.rand(X1.shape[1],1),200,Y1,X1)
print(wtrain)
print("X2shape",X2.shape)
print("y2shape",Y2.shape)
print(MSE(wtrain,X2,Y2))

