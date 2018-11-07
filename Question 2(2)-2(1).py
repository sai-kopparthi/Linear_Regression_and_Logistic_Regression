from sklearn import datasets
import numpy as np
from scipy import sparse
# Question 2
filename = "/home/sai/PycharmProjects/hw1/news20.binary"
x_array,y_array = datasets.load_svmlight_file(filename)

# prediction
def accuracy(xtest,ytest,wtrain):
    s = xtest @ wtrain
    count = 0
    f = np.exp(s)
    f = 1 + f
    f = 1 / f
    print(f)
    y = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if(f[i]<0.5):
            y[i] = -1
        else:
            y[i] = +1
    print(y)
    cam = y.shape[0]
    for i in range (cam):
        if np.sign(y[i]) == np.sign(ytest[i]):
            count = count + 1
    percent = (count / (ytest.shape[0])) * 100
    return percent


#gradient for logistic regression
def deltaa(w,y_array,x_array,n):
    yd = np.diag(y_array)
    yd = np.matrix(yd)
    yd = sparse.csr_matrix(yd)
    q = yd @ x_array
    print("Q shape",q.shape)
    a = q @ w
    qt = q.T
    print("Qt shape",qt.shape)
    m1 = np.exp(a)
    m1 = m1+1
    m1 = 1 / (m1)
    f = (-(1/n))*qt @ m1
    print("m1 shape",m1.shape)
    print("f shape",f.shape)
    print("w shape",w.shape)
    df = ((f)) + w
    print(df.shape)
    return df
print(x_array.shape)
print(y_array.shape)

#wtrain or gradient descent for logistic regression
def grad1(n,e,w,iter,y_array,x_array):
    wa = w
    for i in range(iter):
        g = np.array(deltaa(wa,y_array,x_array,xtest.shape[0]))
        wa = wa - n*deltaa(wa,y_array,x_array,xtest.shape[0])
    return wa

b1 = np.ceil(0.8 * x_array.shape[0])
b1 = b1.astype(int)
print(b1)
xtrain = x_array[1:b1+1,:]
ytrain = y_array[1:b1+1]
xtest = x_array[b1:,:]
ytest = y_array[b1:]
wtrain = grad1(0.1,0.01,np.random.rand(xtest.shape[1],1),10,ytrain,xtrain)
print(wtrain)
print(xtrain.shape)
print(ytrain.shape)
print(accuracy(x_array[b1:,:],y_array[b1:],wtrain))