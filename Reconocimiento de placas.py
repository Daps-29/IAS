import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

placa = []
image = cv2.imread("auto02.jpg",1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray, (3,3))
#cv2.imshow('placagray', gray)

canny = cv2.Canny(gray,150,200)
canny = cv2.dilate(canny, None,iterations=1)
#cv2.imshow('placablur', canny)

cnts,_=cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    area = cv2.contourArea(c)
    х,у,w,h = cv2.boundingRect(c)
    epsilon = 0.09*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP (c, epsilon, True)
    #print(у)
    if len(approx)==4 and area>9000:
        print('area=', area)
        #cv2.drawContours (image, [c],e, (e,255,0), 2)
        aspect_ratio = float(w)/h
        if aspect_ratio>2.0:
            placa = gray[у:у+h,х:х+w] 
            text=pytesseract.image_to_string(placa, config='--psm 11')
            print('text=', text)
            cv2.imshow('placa', placa)
            cv2.moveWindow('placa',780,10)
            cv2.rectangle(image,(х,у),(х+w,у+h),(0,255,0),3)
            cv2.putText(image, text, (х-20,у-10),1,2.2, (0,255,0),3)
            patron = re.compile('[a-zA-Z]{3}-[0-9]{3}')

cv2.imshow('Image', image)
cv2.imshow('Canny', canny)
cv2.moveWindow('Image',45,10)
cv2.waitKey(0)
