#importing required libraries
import cv2
import numpy as np
import handTrackingModule as htm
import time
import autopy #for controlling mouse

#widthCamera, heightCamera
wCam, hCam = 640,480

cap = cv2.VideoCapture(0) #initializing camera 

#setting width and height of the frame # propId for width is 3 and 4 for height 
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    success, img = cap.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
