import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time 
import tensorflow

cap =cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model2/keras_model.h5","Model2/labels.txt")

offset=20
imgsize = 300
folder = 'Data/C'
counter = 0
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

while True:
    success,img = cap.read()
    imgOut = img.copy()
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
            prediction , index = classifier.getPrediction(imgWhite,draw= False)
            print(prediction,index)
        
        else:
            k =imgsize / width
            hCal = math.ceil(k*height) 
            imgResize = cv2.resize(imgCrop,(imgsize,hCal))
            hGap = math.ceil((imgsize - hCal)/2)
            imgWhite[hGap:hCal +hGap,:] = imgResize
            prediction , index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        cv2.rectangle(imgOut,(x-offset,y-offset-50),(x-offset + 90 ,y-offset),(255,0,255),cv2.FILLED)
        cv2.putText(imgOut,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOut,(x-offset,y-offset),(x+w+offset  ,y+ h+ offset),(255,0,255),4)

        # cv2.imshow('Crop Image',imgCrop)
        cv2.imshow('Image White',imgWhite)

    # cv2.putText(imgOut, labels[index])
    cv2.imshow('Image',imgOut)
    cv2.waitKey(1)
    
