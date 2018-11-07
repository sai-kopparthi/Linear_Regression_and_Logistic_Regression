from sklearn import datasets
import numpy as np
from scipy import sparse
filename = "/home/sai/PycharmProjects/hw1/a.txt"
X,Y = datasets.load_svmlight_file(filename)
from scipy import sparse
x_array = sparse.csr_matrix.todense(X)
x_array = x_array/x_array.max(axis=0)
n = x_array.shape[1]
print(n)
y_array = Y
y_array = y_array/np.max(y_array)
y = np.matrix(y_array)

#question 1(1)
#delta f
def deltag(wt,y_array,x_array,n):
    y = np.matrix(y_array)
    yt = y.T
    a = x_array @ wt
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

w_0 = np.random.rand(12,1)
w = grad(0.001,0.001,w_0,200,y_array,x_array)
w1 = grad(0.001,0.001,w_0,200,y_array,x_array)
w2 = grad(0.001,0.001,w_0,200,y_array,x_array)
print("W for 0.001",w)
print("W for 0.0001",w1)
print("W for 0.00001",w2)

#Question 1(2)
#Mean square estimation
def MSE(wttrain, xtest, yttest):
    pred = xtest @ wttrain
    error = pred - yttest
    sumsq = magnitude(error)
    ntest = yttest.shape[0]
    sums = sumsq/ntest
    print("Final MSE", sums)
    return sums

#Fold 1
ef = x_array.shape[0]/5
b = np.ceil(ef)
b = b.astype(int)
print(b)
a = x_array.shape[0]-b
a = a.astype(int)
print(a)
x = x_array[b+1:,:]
y = y_array[b+1:]
y = np.matrix(y)
wtrain=grad(0.00001,0.01,np.random.rand(12,1),200,y,x)
print(wtrain)
xtest=x_array[1:b+1,:]
ytest=y_array[1:b+1]
ytest=np.matrix(ytest)
ytest=ytest.T
fold1 = MSE(wtrain,xtest,ytest)
print("Fold1  MSE ",fold1)

#Fold 2
a1 = x_array[1:b+1,:]
b1 = y_array[1:b+1]
b1 = np.matrix(b1)
a2 = x_array[b*2:,:]
b2 = y_array[b*2:]
b2 = np.matrix(b2)
print("a1",a1.shape)
print("b1",b1.shape)
print("a2",a2.shape)
print("b2",b2.shape)
x = np.concatenate((a1,a2),axis=0)
print(xtest.shape)
y = np.concatenate((b1,b2),axis=1)
print(ytest.shape)
wtrain = grad(0.00001,0.01,np.random.rand(12,1),200,y,x)
print(wtrain)
xtest=x_array[b+1:2*b+1,:]
ytest=y_array[b+1:2*b+1]
ytest=np.matrix(ytest)
ytest=ytest.T
fold2 = MSE(wtrain,xtest,ytest)
print("Fold 2 ",fold2)

#Fold 3
a1 = x_array[1:((b*2)+1),:]
b1 = y_array[1:((b*2)+1)]
b1 = np.matrix(b1)
a2 = x_array[b*3:,:]
b2 = y_array[b*3:]
b2 = np.matrix(b2)
print("a1",a1.shape)
print("b1",b1.shape)
print("a2",a2.shape)
print("b2",b2.shape)
x = np.concatenate((a1,a2),axis=0)
print(xtest.shape)
y = np.concatenate((b1,b2),axis=1)
print(ytest.shape)
wtrain = grad(0.00001,0.01,np.random.rand(12,1),200,y,x)
print(wtrain)
xtest=x_array[b*2:(b*3)+1,:]
ytest=y_array[b*2:(b*3)+1]
ytest=np.matrix(ytest)
ytest=ytest.T
fold3 = MSE(wtrain,xtest,ytest)
print("Fold 3 ",fold3)

# fold 4
a1 = x_array[1:((b*3)+1),:]
b1 = y_array[1:((b*3)+1)]
b1 = np.matrix(b1)
a2 = x_array[b*4:,:]
b2 = y_array[b*4:]
b2 = np.matrix(b2)
print("a1",a1.shape)
print("b1",b1.shape)
print("a2",a2.shape)
print("b2",b2.shape)
x = np.concatenate((a1,a2),axis=0)
print(xtest.shape)
y = np.concatenate((b1,b2),axis=1)
print(ytest.shape)
wtrain = grad(0.00001,0.01,np.random.rand(12,1),200,y,x)
print(wtrain)
xtest=x_array[b*3:(b*4)+1,:]
ytest=y_array[b*3:(b*4)+1]
ytest=np.matrix(ytest)
ytest=ytest.T
fold4 = MSE(wtrain,xtest,ytest)
print("Fold 4 ",fold4)


# fold 5

x=x_array[1:a,:]
y=y_array[1:a]
y = np.matrix(y)
wtrain=grad(0.00001,0.01,np.random.rand(12,1),200,y,x)
print(wtrain)
xtest=x_array[a:,:]
ytest=y_array[a:]
ytest=np.matrix(ytest)
ytest=ytest.T
fold5 = MSE(wtrain,xtest,ytest)
print("Fold5  MSE ",fold5)
fold = fold1+fold2+fold3+fold4+fold5
avgMSEfold=(fold/5)
print("Avg MSE",avgMSEfold)



