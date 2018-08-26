# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 09:54:30 2018

@author: AKMH
"""
import cv2
import numpy as np
#import numpy as np
#                      1 for external cam
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
import pandas as pd
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import classification_report
#from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from datetime import datetime

dataset = pd.read_csv('shirt_data.csv').as_matrix()

classifier = DecisionTreeClassifier()
X_train = dataset[1:,1:] 
train_label = dataset[1: ,0]
classifier.fit(X_train, train_label)
#xtest = dataset[101,1:]
while True:
    ret,frame = cap.read()
#    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
#    cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('background.jpg',frame)
        break
#cap.release()

cv2.destroyAllWindows()
bk = cv2.imread('background.jpg',0)
while True:
    ret , frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = gray - bk
    bilFilter = cv2.bilateralFilter(img,50,100,100)
    gausBlur1 = cv2.GaussianBlur(bilFilter, (5,5),0)
    blur = cv2.GaussianBlur(gausBlur1,(5,5),0)
#    edge0 = cv2.Canny(gray,100,150)
    t0 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)[1]
    
    img = cv2.resize(img,(50,50))
    for i in range(0,50):
        for j in range(0,50):
            if img[i,j]>25:
                img[i,j]=255;
#    gray=[np.float64(i) for i in gray]
#    noise=np.random.randn(*gray[1].shape)*10
#    noisy=[i+noise for i in gray]
#    noisy=[np.uint8(np.clip(i,0,255)) for i in noisy]
#    dst=cv2.fastNlMeansDenoisingMulti(noisy,2,5,None,4,7,35)
    cv2.imshow('gray',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    #print("contour len:",len(contours))
    #img = cv2.drawContours(img, contours,3,(0,255,0),3)
    #cv2.imshow('gray',img)
    #cv2.waitKey(0)
cv2.destroyAllWindows()
arr = np.array(img)
shape = arr.reshape(1,50*50)

# make a 1-dimensional view of arr
flat_arr = arr.ravel()
pre = classifier.predict(shape)
print(pre)
if(pre == ['S']):
    print('Collar : 14\" - 14.5\" ','\n','waist : 34\" - 37\"')
if(pre == ['XL']):
    print('Collar : 17\" - 17.5\" ','\n','waist : 36\" - 38\"')
#print(shape)