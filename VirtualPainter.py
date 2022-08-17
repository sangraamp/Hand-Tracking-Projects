import cv2
import numpy as np
import time
import platform
import os
import HandTrackingModule as htm

#############################
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
#############################

folderPath = 'Paint'
myList = os.listdir(folderPath)
if platform.system() == 'Darwin':
    myList.pop(0) # DS store not needed in mac
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[2]
drawColor = (255, 0, 255) # pink/purple by default

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), dtype=np.uint8)

detector = htm.HandDetector(detectionCon=0.85)

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()

        # 4. If two fingers are up (selection mode) then select, don't draw
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('selection')

            if y1 < 125:
                if 120 < x1 < 250:
                    header = overlayList[2] # pink/purple
                    drawColor = (255, 0, 255)
                elif 440 < x1 < 560:
                    header = overlayList[3] # blue
                    drawColor = (255, 0, 0)
                elif 760 < x1 < 880:
                    header = overlayList[0] # green
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1180:
                    header = overlayList[1] # eraser
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If index finger is up - drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print('drawing')

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, ERASER_THICKNESS)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, ERASER_THICKNESS)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, BRUSH_THICKNESS)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, BRUSH_THICKNESS)

            xp, yp = x1, y1

        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header

    cv2.imshow('Image', img)
    cv2.waitKey(1)