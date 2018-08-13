# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Calculation of derivative of second layer parameters is slightly off. So result may degrade some times. Still working on it!

import os
import random
import time
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit
from scipy.optimize import minimize
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

pd.set_option('max_columns',None)
pd.options.display.width = 2000
np.set_printoptions(threshold=np.nan)
np.set_printoptions(suppress=True)

mpl.rcParams['agg.path.chunksize'] = 10000
%matplotlib inline

def printruntime(label,startTime):
    print(label, (time.time()-startTime))
    
def show_data_sample():
    fig,ax = plt.subplots(5,2)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    count = 0
    for row in ax:
            for subplot in row:
                subplot.imshow(digits_bunch.images[count])
                subplot.set_title(str(digits_bunch.target[count])+","+str(outputs[count]))
                count = count + 1
                
    plt.show()
                
def form_output_vectors():
    outputs = np.zeros((digits_bunch.data.shape[0],K))
    outputs[range(outputs.shape[0]),digits_bunch.target] = 1
    return outputs

runtime = time.time()
K = 10
features = 64
digits_bunch = load_digits(n_class=K)
m = digits_bunch.data.shape[0]
#print(digits_bunch.DESCR) # to see full description
#print(type(digits_bunch.target[0]),type(digits_bunch.target_names[0]),type(digits_bunch.data[0][0]),type(digits_bunch.images[0][0][0]))
outputs = form_output_vectors() #(1797, 10)
#show_data_sample()

#let's use NN with 3 layers first. layer1(i/p) = 64 units, layer2(hidden) = 100 units, layer3(o/p) = 10 units
#excluding bias
l1_size = features
l2_size = 100
l3_size= K
theta1 = np.random.uniform(0,3,(l1_size+1, l2_size)) #(65,100) #(l1_size+1, l2_size)
theta2 = np.random.uniform(0,3,(l2_size+1,l3_size)) #(101,10)
xy = np.column_stack((digits_bunch.data,outputs)) #(1797,64+10)
x_tr,x_te = train_test_split(xy,train_size=0.7)
y_tr = x_tr[:,64:]
y_te = x_te[:,64:]
x_tr = x_tr[:,0:64]
x_te = x_te[:,0:64]

del xy

print("Train Test sets",x_tr.shape,y_tr.shape,x_te.shape,y_te.shape)

def checkGrad(x,y,theta1,theta2):
    eps = 0.0000001 #0.00001    
    theta = np.concatenate((np.asarray(theta1).flatten(),np.asarray(theta2).flatten()))
    gradApprox = np.zeros((len(theta,)))
    thetaPlus = np.copy(theta)
    thetaMinus = np.copy(theta)
    print("Total iterations to be made",len(theta))
    for i in range(len(theta)):
        if(i % 100 == 0):
            print("iteration",i)
        if(i != 0):
            thetaPlus[i-1] = thetaPlus[i-1]-eps
            thetaMinus[i-1] = thetaMinus[i-1]+eps
        thetaPlus[i] = theta[i]+eps
        thetaMinus[i] = theta[i]-eps
        cost1,grad1 = cost(thetaPlus,x,y,theta1.shape,theta2.shape)
        cost2,grad2 = cost(thetaMinus,x,y,theta1.shape,theta2.shape)
        gradApprox[i] = (cost1 - cost2)/(2*eps)
        
    return gradApprox
    #cost_minus,grad1 = cost(np.concatenate((np.asarray(theta1-eps).flatten(),np.asarray(theta2-eps).flatten())),x,y,theta1.shape,theta2.shape)
    #cost_plus,grad2 = cost(np.concatenate((np.asarray(theta1+eps).flatten(),np.asarray(theta2+eps).flatten())),x,y,theta1.shape,theta2.shape)
    #cost_delta = (cost_plus-cost_minus)/2*eps
    #print("cost delta",cost_delta)

def cost(flatThetas,x,y,theta1size,theta2size):
    theta1 = np.matrix(np.reshape(flatThetas[0:theta1size[0]*theta1size[1]],theta1size))
    theta2 = np.matrix(np.reshape(flatThetas[theta1size[0]*theta1size[1]:],theta2size))
    outputs = y
    m = x.shape[0]
    #forward propagation
    a1 = x #(1797, 64)
    a1 = np.column_stack((np.ones(m,),a1)) #(1797,65)
    a2 = expit(a1.dot(theta1)) #(1797,100)
    a2 = np.column_stack((np.ones(m,),a2)) #(1797,101)
    a3 = expit(a2.dot(theta2)) #(1797,10)
    a3[a3==1] = 0.999999
    res1 = np.multiply(outputs,np.log(a3)) #(1797,10) .* (1797,10) 
    res2 = np.multiply(1-outputs,np.log(1-a3))
    lamda = 0.5
    cost = (-1/m)*(res1+res2).sum(axis=1).sum() + lamda/(2*m)*(np.square(theta1[1:,:]).sum(axis=1).sum() + np.square(theta2[1:,:]).sum(axis=1).sum())
    #print("Cost",cost)
    
    #Back propagation
    delta3 = a3 - outputs
    delta2 = np.multiply(delta3.dot(theta2.T),np.multiply(a2,1-a2)) #(1797,10) * (10,101) = (1797,101)
    D1 = (a1.T.dot(delta2[:,1:])) #(65, 1797) * (1797,100) = (65,100)
    D1[0,:] = 1/m * D1[0,:]
    D1[1:,:] = 1/m * (D1[1:,:] + lamda*theta1[1:,:])
    D2 = (a2.T.dot(delta3)) #(101,1797) * (1797, 10) = (101,10)
    D2[0,:] = 1/m * D2[0,:]
    D2[1:,:] = 1/m * (D2[1:,:] + lamda*theta2[1:,:]) #something wrong in D2 calculation steps...
    #print(theta1.shape,theta2.shape,D1.shape,D2.shape)
    return cost,np.concatenate((np.asarray(D1).flatten(),np.asarray(D2).flatten())) #last 1010 wrong values

#def iter_callback(xk,opt_res):
#    print("callback",opt_res.nit)

grad_check = False
#This is a time consuming process. Should only be used while debugging. grad_check = False
if(grad_check):
    print("Checking gradient:")
    c,grad = cost(np.concatenate((np.asarray(theta1).flatten(),np.asarray(theta2).flatten())),x_tr,y_tr,theta1.shape,theta2.shape)
    grad_approx = checkGrad(x_tr,y_tr,theta1,theta2)
    print("Non zero in grad",np.count_nonzero(grad),np.count_nonzero(grad_approx))
    tup_grad = np.nonzero(grad)
    print("Original\n",grad[tup_grad[0][0:20]])
    print("Numerical\n",grad_approx[tup_grad[0][0:20]])
    wrong_grads = np.abs(grad-grad_approx)>0.1
    print("Max diff:",np.abs(grad-grad_approx).max(),np.count_nonzero(wrong_grads),np.abs(grad-grad_approx)[0:6500].max())
    print(np.squeeze(np.asarray(grad[wrong_grads]))[0:20])
    print(np.squeeze(np.asarray(grad_approx[wrong_grads]))[0:20])
    where_tup = np.where(wrong_grads)
    print(where_tup[0][0:5],where_tup[0][-5:])

print("Starting optimization")
#Newton-CG worked cost=5. #TNC
opt_res = minimize(cost,np.concatenate((np.asarray(theta1).flatten(),np.asarray(theta2).flatten())),(x_tr,y_tr,theta1.shape,theta2.shape),method = 'TNC',jac=True,options={'maxiter':1000,'disp':True}) #,callback=iter_callback

print("Optimization is_success?",opt_res.success, opt_res.message)
res = opt_res.x
c,grad = cost(res,x_te,y_te,theta1.shape,theta2.shape)
print("final cost",c)
del grad

def guess_digits(flatThetas,x,y,theta1size,theta2size):
    theta1 = np.matrix(np.reshape(flatThetas[0:theta1size[0]*theta1size[1]],theta1size))
    theta2 = np.matrix(np.reshape(flatThetas[theta1size[0]*theta1size[1]:],theta2size))
    m = x.shape[0]
    #forward propagation
    a1 = x #(1797, 64)
    a1 = np.column_stack((np.ones(m,),a1)) #(1797,65)
    a2 = expit(a1.dot(theta1)) #(1797,100)
    a2 = np.column_stack((np.ones(m,),a2)) #(1797,101)
    a3 = expit(a2.dot(theta2)) #(1797,10)
    #a3[a3==1] = 0.9999
    h = np.argmax(a3,axis=1) #(m,1)
    op = np.zeros((x.shape[0],1))
    for i,row in enumerate(y):
        op[i,0] = np.where(row==1)[0]
    
    print("success rate",(np.count_nonzero(h==op)/len(h))*100)
    fig,ax = plt.subplots(10,3)
    plt.title("Using own implementation")
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.tight_layout()
    count = 0
    for row in ax:
            for subplot in row:
                subplot.imshow(np.reshape(x[count],(8,8)))
                subplot.set_title("Real:"+str(op[count]).replace("[","").replace("]","")+",Guess:"+str(h[count][0]).replace("[","").replace("]",""))
                count = count + 1
                
    plt.show()
    
    #df_res = pd.DataFrame({'pred':h,'original':op})
    #display(df_res)

#Now let's guess digits
guess_digits(res,x_te,y_te,theta1.shape,theta2.shape)

# Using MLPClassifier
print("Using MLPClassifier...")
xy = np.column_stack((digits_bunch.data,digits_bunch.target)) #(1797,65)
x_tr,x_te = train_test_split(xy,train_size=0.7)
y_tr = x_tr[:,-1]
y_te = x_te[:,-1]
x_tr = x_tr[:,0:64]
x_te = x_te[:,0:64]
print(x_te.shape,x_te[0:2].shape)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100), random_state=1,max_iter=300,verbose=True)
clf.fit(x_tr,y_tr)
y = clf.predict(x_te)
print("success rate",(np.count_nonzero(y==y_te)/len(y))*100)
fig,ax = plt.subplots(10,3)
plt.title("Using MLPClassifier")
fig.set_figheight(20)
fig.set_figwidth(20)
fig.tight_layout()
count = 0
for row in ax:
        for subplot in row:
            subplot.imshow(np.reshape(x_te[count],(8,8)))
            subplot.set_title("Real:"+str(y_te[count])+",Guess:"+str(y[count]))
            count = count + 1

plt.show()

    
printruntime("Total Runtime = ",runtime)
