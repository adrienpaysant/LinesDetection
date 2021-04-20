import os
import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt

def edgeDetectAndShowCanny(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'hsv')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def edgeDetectAndShowHough(imagePath):
    #base image
    img = cv2.imread(imagePath)
    #gray image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    showAndWait('Gray',gray)
    #edges
    edges = cv2.Canny(img,75,200,apertureSize=3,L2gradient=True)
    showAndWait('Edges',edges)
    #hough
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,45)
    assert circles is not None, '-no circle found-'
    print('circle(x,y,radius',circles)
    showAndWait("Circles",circles)

    #draw circle on image
    circles=np.uint16(np.around(circles))
    for i in circles[0,:]:
        #outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        #center circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    showAndWait('Circle Detection',img)

def shapeDetection(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,127,255,1)

    contours,h = cv2.findContours(thresh,1,2)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print (len(approx))
        if len(approx) > 15:
            print ("circle")
            cv2.drawContours(img,[cnt],0,(0,255,255),-1)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def showAndWait(string, img) :
    cv2.imshow(string, img)
    cv2.waitKey(0)

def imgCaract(imagePath):
    img=cv2.imread(imagePath)
    cv2.imshow(imagePath,img)#affichage image
    print("\n******\n"+"INFOS IMAGE : "+imagePath)
    h,w,c=img.shape
    print("DEFINITION : w : %d, h : %d, channel : %d"%(w,h,c))
    print("TAILLE : ",img.size)
    print("TYPE DONNEES : ",img.dtype)
    print("MINIMUM : ",np.amin(img)," MAXI : ",np.amax(img))
    print("MOYENNE : ",np.mean(img))
    print("ECART TYPE",np.std(img))
    print("MODE : ",np.argmax(np.bincount(img.flatten())))
    print("******")