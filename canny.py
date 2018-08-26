# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 02:45:37 2018

@author: AKMH
"""

import cv2

cam = cv2.VideoCapture(0)
cam.set(3,320)
cam.set(4,240)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret , frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bilFilter = cv2.bilateralFilter(gray,50,100,100)
    gausBlur1 = cv2.GaussianBlur(bilFilter, (5,5),0)
    blur = cv2.GaussianBlur(gausBlur1,(5,5),0)
    edge0 = cv2.Canny(gray,100,150)
    t0 = cv2.threshold(edge0,127,255,cv2.THRESH_BINARY_INV)[1]
#    for (x,y,w,h) in face
    cv2.imshow('gray',t0)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
#print("contour len:",len(contours))
#img = cv2.drawContours(img, contours,3,(0,255,0),3)
#cv2.imshow('gray',img)
#cv2.waitKey(0)
cv2.destroyAllWindows()