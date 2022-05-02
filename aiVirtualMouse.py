#importing required libraries
from enum import auto
from logging import logThreads
from stat import FILE_ATTRIBUTE_INTEGRITY_STREAM
import cv2
import numpy as np
import handTrackingModule as htm
import time
import autopy #for controlling mouse

####################
# 1. Find the hand landmarks
# 2. Get the tip of the index and the middle finger 
#     [ when only index finger is up - mouse will move,
#     when index finger and middle finger is up and they are together, then it will be in the clicking mode]
# 3. Check which finger among index and middle fingers is/are up.
# 4. Only Index Finger : Moving mode:
#     4.1 Convert the coordinates to get the correct positioning
#     4.2. Smoothen the values so that it does not flicker a lot 
#     4.3. Move the mouse
# 5. Both index and Middle fingers are up : Clicking mode
#     5.1 Find distance between both the fingers: Click the mouse if distance is short
# 6. Frame Rate 
# 7. Display
###################


wCam, hCam = 640,480
wScreen,hScreen = autopy.screen.size()
frameReduction = 100
smoothning = 6
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0
pTime = 0

cap = cv2.VideoCapture(0) #initializing camera 

#setting width and height of the frame # propId for width is 3 and 4 for height 
cap.set(3,wCam)
cap.set(4,hCam)



detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()

    # 1. Find the hand landmarks
    img = detector.findHands(img)
    lmList,bbox = detector.findPosition(img)
    
    # 2. Get the tip of the index and the middle finger 
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:] # for index finger
        x2,y2 = lmList[12][1:] # for middle finger
        # print(x1,y1,x2,y2)

        # 3. Check which finger among index and middle fingers is/are up.
        fingers = detector.fingersUp()
        print(fingers)
        cv2.rectangle(img,(frameReduction,frameReduction),(wCam-frameReduction,hCam-frameReduction),
                        (255,0,255),2)
        # 4. Only Index Finger : Moving mode:
        if fingers[1] == 1 and fingers[2]== 0:
            #means index finger is up and middle finger is down
        
            #4.1 Convert the coordinates to get the correct positioning
            cv2.rectangle(img,(frameReduction,frameReduction),(wCam-frameReduction,hCam-frameReduction),
                            (255,0,255),2)
            x3 = np.interp(x1,(frameReduction,wCam-frameReduction),(0,wScreen))
            y3 = np.interp(y1,(frameReduction,hCam-frameReduction),(0,hScreen))
            
            #4.2. Smoothen the values so that it does not flicker a lot 
            currLocX = prevLocX + (x3-prevLocX)/smoothning
            currLocY = prevLocY + (y3-prevLocY)/smoothning

            #4.3. Move the mouse
            autopy.mouse.move(wScreen-currLocX,currLocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            
            prevLocX, prevLocY = currLocX, currLocY
        
        # 5. Both index and Middle fingers are up : Clicking mode
        if fingers[1] == 1 and fingers[2]== 1:
            #means both index and middle fingers are up
            # 5.1 Find distance between both the fingers: Click the mouse if distance is short
            length,img,lineInfo = detector.findDistance(8,12,img)
            print(length)
            if length<40:
                #click detected
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                #click
                autopy.mouse.click()

            




    # 6. Frame Rate 
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    # 7. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
