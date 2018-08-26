# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 03:25:06 2018

@author: Munavar Hussain
"""
import cv2
#import numpy as np
#                      1 for external cam
cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)
while True:
    ret,frame = cap.read()
#    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
#    cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('background.jpg',frame)
        break
cap.release()
cv2.destroyAllWindows()