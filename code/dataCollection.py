import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time 

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset=20
imgsize = 300
folder = 'Data/Z'
counter = 0

while True:
    success,img = cap.read()
    hands,img=detector.findHands(img)

    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]
        imgCropShape = imgCrop.shape
        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        # imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop

        height = h  + offset
        width = w + offset 

        aspectRatio = height /width
        if aspectRatio > 1 :
            k =imgsize / height
            wCal = math.ceil(k*width) 
            imgResize = cv2.resize(imgCrop,(wCal,imgsize))
            wGap = math.ceil((imgsize - wCal)/2)
            imgWhite[:,wGap:wCal +wGap] = imgResize
        
        else:
            k =imgsize / width
            hCal = math.ceil(k*height) 
            imgResize = cv2.resize(imgCrop,(imgsize,hCal))
            hGap = math.ceil((imgsize - hCal)/2)
            imgWhite[hGap:hCal +hGap,:] = imgResize

        cv2.imshow('Crop Image',imgCrop)
        cv2.imshow('Image White',imgWhite)
    cv2.imshow('Image',img)
    key =cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
