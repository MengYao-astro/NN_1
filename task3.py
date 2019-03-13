#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 09:41:23 2019

@author: meng
"""


import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix

#read in data
train_in=np.genfromtxt('train_in.csv',delimiter=',')
train_out=np.genfromtxt('train_out.csv',delimiter=',')
test_in=np.genfromtxt('test_in.csv',delimiter=',')
test_out=np.genfromtxt('test_out.csv',delimiter=',')


#drag out 0&1
train_in0=[train_in[x,:] for x in np.where(train_out==0)][0]
train_in1=[train_in[x,:] for x in np.where(train_out==1)][0]
#feature : number of peaks in the middle rows 7th & 8th
#for "0", generally it should have 2 peak in a single row, so 4 peaks in 2 rows
#for "1", generally it should have 1 peak in a single row, so 2 peaks in 2 rows.
# digit 0
x0=[]
for i in np.arange(len(train_in0)):
    train_in0mr7=train_in0[i,:][16*7:16*8]
    train_in0mr8=train_in0[i,:][16*8:16*9]
    #extend the edges add '-1' at the end, insert -1 at the begining.
    #it will be better to detect the peaks
    train_in0mr7=np.insert(np.append(train_in0mr7,-1),0,-1)
    train_in0mr8=np.insert(np.append(train_in0mr8,-1),0,-1)
    x0.append(len(find_peaks(train_in0mr7)[0])+len(find_peaks(train_in0mr8)[0]))
x0=np.array(x0)
#digit 1
x1=[]
for i in np.arange(len(train_in1)):
    train_in1mr7=train_in1[i,:][16*7:16*8]
    train_in1mr8=train_in1[i,:][16*8:16*9]
    #extend the edges add '-1' at the end, insert -1 at the begining.
    train_in1mr7=np.insert(np.append(train_in1mr7,-1),0,-1)
    train_in1mr8=np.insert(np.append(train_in1mr8,-1),0,-1)
    x1.append(len(find_peaks(train_in1mr7)[0])+len(find_peaks(train_in1mr8)[0]))
x1=np.array(x1)
#plot histogram
fig=plt.figure()
ax1=plt.subplot(1,2,1)
ax1.hist(x0,bins=np.arange(1.5,7.5,1),label='peaks of 0')
ax1.hist(x1,bins=np.arange(0.5,6.5,1),label='peaks of 1')
ax1.legend(loc='best')
ax1.set_title('statistical histogram')

#Bayes
xall=np.hstack((x0,x1))
P3=sum(xall==3)/len(xall)      #P3=P3_C0*PC0+P3_C1*PC1
PC0=len(train_in0)/(len(train_in0)+len(train_in1))
PC1=len(train_in1)/(len(train_in0)+len(train_in1))
P3_C0=sum(x0==3)/len(train_in0)
P3_C1=sum(x1==3)/len(train_in1)
# hence
PC0_3=P3_C0*PC0/P3     #when x=3, posteriors of the digit being 0
print('P(C0|X=3)=',PC0_3)
PC1_3=P3_C1*PC1/P3     #when x=3, posteriors of the digit being 1
print('P(C1|X=3)=',PC1_3)
PC0_2=1                #when x=2, the digit is 0
PC1_g4=1               #when x>=4  the digit is 1
                       #otherwise = 0  
#probability histogram
def prob0(vx):
    if vx < 2.5 :
        return 0
    if vx > 3.5 :
        return 1
    else :
        return PC0_3
def prob1(vx):
    if vx < 2.5 :
        return 1
    if vx > 3.5 :
        return 0
    else :
        return PC1_3
#plt.figure()
xvalue = np.linspace(0, 7, 1000)
y0 = np.array([])
y1 = np.array([])
for v in xvalue:
    y0 = np.append(y0,prob0(v))
    y1 = np.append(y1,prob1(v))
ax2=plt.subplot(1,2,2)
ax2.plot(xvalue,y0,label='P(C0|x)')
ax2.plot(xvalue,y1,label='P(C1|x)')
ax2.legend(loc='best')
ax2.set_title('posterior probability distribution')
plt.show()                     
#expected loss
#because generally, assuming a number begins wit 1, if we say '1' is '0', we will lose the value on digit level.
L00=0
L01=1 
L10=4 
L11=0
#if x=3 , we say '0'. loss=
loss0=PC0_3*L00+PC1_3*L10
#if x=3 , we say '1'. loss=
loss1=PC0_3*L01+PC1_3*L11
#Discriminant Function P0-P1
def disfy(vx):         # define y= y(0)-y)(1)
    if vx > 3.5 :
        return 1 # probability of saying it is 0 is 1, y(0)=1, y(1)=0
    if vx < 2.5 :
        return -1 # y(0)=0, y(1)=1
    if vx == 3:
        if loss0>loss1 :     # consider the risk, if the loss of saying it is 0 is larger, then we would better say it is 1
            return -1         # so y(0)=0 y(1)=1 
        else:  
            return 1         # otherwise the probability is 0
    
#test
#drag out 0&1
test_in0=[test_in[x,:] for x in np.where(test_out==0)][0]
test_in1=[test_in[x,:] for x in np.where(test_out==1)][0]
# digit 0
testresult0=[]
for i in np.arange(len(test_in0)):
    test_in0mr7=test_in0[i,:][16*7:16*8]
    test_in0mr8=test_in0[i,:][16*8:16*9]
    #extend the edges add '-1' at the end, insert -1 at the begining.
    test_in0mr7=np.insert(np.append(test_in0mr7,-1),0,-1)
    test_in0mr8=np.insert(np.append(test_in0mr8,-1),0,-1)
    x=len(find_peaks(test_in0mr7)[0])+len(find_peaks(test_in0mr8)[0])
    if disfy(x) > 0 :
        testresult0.append(0)
    if disfy(x) < 0 :
        testresult0.append(1)
testresult0=np.array(testresult0)
#digit 1

testresult1=[]
for i in np.arange(len(test_in1)):
    test_in1mr7=test_in1[i,:][16*7:16*8]
    test_in1mr8=test_in1[i,:][16*8:16*9]
    #extend the edges add '-1' at the end, insert -1 at the begining.
    test_in1mr7=np.insert(np.append(test_in1mr7,-1),0,-1)
    test_in1mr8=np.insert(np.append(test_in1mr8,-1),0,-1)
    x=len(find_peaks(test_in1mr7)[0])+len(find_peaks(test_in1mr8)[0])
    if disfy(x) > 0 :
        testresult1.append(0)
    if disfy(x) < 0 :
        testresult1.append(1)
testresult1=np.array(testresult1)
#accuracy
accuracy=1-(sum(testresult0==1)+sum(testresult1==0))/(len(testresult0)+len(testresult1))
print('accuracy=',accuracy)
#confusion matrix
confusion_in=np.append(np.zeros(len(test_in0)),np.ones(len(test_in1)))
confusion_out=np.append(testresult0,testresult1)
#confusion_matrix(confusion_in,confusion_out)
