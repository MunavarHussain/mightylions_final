# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:37:30 2018

@author: AKMH
"""

import cv2

cam = cv2.VideoCapture(0)
cam.set(3,320)
cam.set(4,240)
bk = cv2.imread('background.jpg',0)
while True:
    ret , frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    img = gray - bk
    bilFilter = cv2.bilateralFilter(img,50,100,100)
    gausBlur1 = cv2.GaussianBlur(bilFilter, (5,5),0)
    blur = cv2.GaussianBlur(gausBlur1,(5,5),0)
#    edge0 = cv2.Canny(gray,100,150)
    t0 = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)[1]
    
    img = cv2.resize(img,(200,200))
    for i in range(0,200):
        for j in range(0,200):
            if img[i,j]>25:
                img[i,j]=255;
    
    cv2.imshow('gray',img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
#print("contour len:",len(contours))
#img = cv2.drawContours(img, contours,3,(0,255,0),3)
#cv2.imshow('gray',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()