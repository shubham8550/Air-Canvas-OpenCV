import cv2
import time
import HandTrackingModule as htm
import numpy as np
import os



overlayList = [] # to store all images

brushThickness =15
eraserThickness = 100
drawColor=(0,0,255)

xp, yp = 0, 0  # Previous  points
imgCanvas = np.zeros((720,1280,3),np.uint8)

# loading imagegs from assets
folderPath = "assets"
myList = os.listdir(folderPath)
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
header = overlayList[1]
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.50,maxHands=1) #making obj

while True:
    #import image
    success, img = cap.read()
    img=cv2.flip(img,1)
    # if not success:
    #     continue

    # find hands Landmarks
    img = detector.findHands(img)
    lmList ,bbox =detector.findPosition(img,draw=False)

    if len(lmList) != 0:
        #print(lmList)
        x1,y1=lmList[8][1],lmList[8][2] # ttip of index finger
        x2,y2=lmList[12][1],lmList[12][2] # ttip of middle finger

        # check which fingers are up
        fingers = detector.fingersUp()

        # if Selection Mode - Two fingers up
        if fingers[1] and fingers[2]:
            xp,yp= 0,0

            #checking if index finger tip is in menu region
            #and for click
            if y1 < 120:
                if 34 < x1 < 234: # if im clicking at Red brush
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 262 < x1 < 462:# if im clicking at Green brush
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 490 < x1 < 690:# if im clicking at Blue brush
                    header = overlayList[3]
                    drawColor = (255, 0, 0)
                elif 837 < x1 < 988:  # if i m clicking at eraser
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor,cv2.FILLED)  # selection mode is represented as rectangle

        # if drawing Mode  - index finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1),15,drawColor,cv2.FILLED) # drawing mode is represented as circle

            if xp == 0 and yp == 0:  # initially xp and yp will be at 0,0 so it will draw a line from 0,0 to whichever point our tip is at
                xp, yp = x1, y1  # so to avoid that we set xp=x1 and yp=y1
            # till now we are creating our drawing but it gets removed as everytime our frames are updating so we have to define our canvas where we can draw and show also

            #eraser
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp), (x1, y1),drawColor,eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor,brushThickness)  # gonna draw lines from previous coodinates to new positions
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp =x1,y1 #writing history
        else:
            xp,yp = 0, 0

        # merging two windows into one imgcanvas and img

    # 1 converting img to gray
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    # 2 converting into binary image and thn inverting
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)  # on canvas all the region in which we drew is black and where it is black it is cosidered as white,it will create a mask

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)  # converting again to gray bcoz we have to add in a RGB image i.e img

    # add original img with imgInv ,by doing this we get our drawing only in black color
    img = cv2.bitwise_and(img, imgInv)

    # add img and imgcanvas,by doing this we get colors on img
    img = cv2.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:120, 0:1280] = header  # on our frame we are setting our JPG image acc to H,W of jpg images

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
















