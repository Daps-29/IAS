
import cv2
import numpy as np
from rastrador import *   
import time

seguimiento = Rastreador()
cap = cv2.VideoCapture("Avenida.mp4")
deteccion = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold=100)

carI = {}
carO = {}
prueba = {}

while True:
    ret,frame = cap.read()
    #frame = cv2.resize(frame,(1920,1080))
    height = frame.shape[0]
    width = frame.shape[1]
    
    mask = np.zeros((height,width), dtype=np.uint8)
    pts = np.array([[[680,300],[800,300],[1350,730],[754,730]]])
    
    cv2.fillPoly(mask,pts,255)
    zona = cv2.bitwise_and(frame,frame,mask=mask)
    
    areag=([680,350],(830,350),(1350,730),(754,730))
    area3=([680,350],(830,350),(920,400),(690,400))
    areal=([710,490],(1020,490),(1150,600),(720,600))
    area2=([690,400],(920,400),(1020,490),(710,490))
    
    cv2.polylines(frame,[np.array(areag,np.int32)],True,(255,255,0),2)
    cv2.polylines(frame,[np.array(area3,np.int32)],True,(128,0,128),1)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),1)
    cv2.polylines(frame,[np.array(areal,np.int32)],True,(0,255,0),1)
    
    mascara = deteccion.apply(zona)
    
    filtro = cv2.GaussianBlur(mascara,(11,11),0)
    
    _, umbral = cv2.threshold(filtro,50,255,cv2.THRESH_BINARY)
    dila = cv2.dilate(umbral,np.ones((3,3)))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    cerrar = cv2.morphologyEx(dila,cv2.MORPH_CLOSE,kernel)
    
    contornos, _ = cv2.findContours(cerrar,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    detecciones = []
    for cont in contornos:
        area = cv2.contourArea(cont)
        if area > 1800:
            x,y,ancho,alto = cv2.boundingRect(cont)
            
            detecciones.append([x,y,ancho,alto])
            
    info_id = seguimiento.rastreo(detecciones)
    for inf in info_id:
        x,y,ancho,alto,id = inf
        
        cv2.rectangle(frame,(x,y-10),(x+ancho,y+alto),(0,0,255),2)
        cx = int(x+ancho/2)
        cy = int(y+alto/2)
        a2 = cv2.pointPolygonTest(np.array(area2,np.int32),(cx,cy),False)
        if a2 >= 0:
            carI[id] = time.process_time()
        if id in carI:
            cv2.circle(frame, (cx,cy),3,(0,0,255),-1)
            a3 = cv2.pointPolygonTest(np.array(area3,np.int32),(cx,cy),False)
            if a3 >= 0:
                tiempo = time.process_time() - carI[id]
                if tiempo % 1 ==0:
                    tiempo = tiempo +0.323
                if tiempo % 1 !=0:
                    tiempo = tiempo +1.016
                if id not in carO:
                    carO[id] = tiempo
                if id in carO:
                    tiempo = carO[id]
                    vel = 14.3/carO[id]
                    vel = vel * 3.6
                
                cv2.rectangle(frame,(x,y-10),(x+100,y-50),(0,0,255),-1)
                cv2.putText(frame,str(int(vel)) + "KM / H" , (x,y-35),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        cv2.putText(frame,str(id),(x,y-15),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)
    cv2.imshow("Carretera",frame)
    cv2.imshow("Mascara",cerrar)
    key = cv2.waitKey(5)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
        