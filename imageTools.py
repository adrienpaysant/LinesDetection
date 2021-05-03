import os
import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics
import math

WIDTH_SCREEN= GetSystemMetrics(0)
HEIGHT_SCREEN= GetSystemMetrics(1)

############################################################
############################################################
###################Basics Transforms########################
############################################################
############################################################

def hsvSpace(imagePath):
    """ return image form give path on HSV colors"""
    img=cv2.imread (imagePath)
    return  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
############################################################
############################################################
####################Shape Detection#########################
############################################################
############################################################

class Marble:
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        self.distance=-1
    
    def __str__(self):
        return f"center = {self.center} and radius = {self.radius}"    

#https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def getClosestPointOnLine(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    coef = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/coef
    return int(x1+a*dx), int(y1+a*dy)

def getSideOfLine(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if (x2-x1)*(y2-y3) - (y2-y1)*(x2-x3) > 0:
        return False
    print ('A marble Won')
    return True

def getMask(grayImage,lowerValueRange):
    mask = cv2.GaussianBlur(grayImage, (11, 11), 0)
    mask = cv2.inRange(mask, lowerValueRange, 255)
    return mask

def erodeDilate(grayImage,mask):
    res = cv2.bitwise_and(grayImage, grayImage, mask=mask)
    res = cv2.erode(res, None, iterations=3)
    res = cv2.dilate(res, None, iterations=1)
    return res

def getX(elem):
    return elem.center[0]

def shapeDetection(img,lowerValueRange,radiusArrivalObject):
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = getMask(grayImage,lowerValueRange)
    result = erodeDilate(grayImage,mask)

    newImage = img
    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImage, contours, -1, (255, 0, 0), 5)
    # showAndWait("test",newImage)
    marblesSet = set()
    for cnt in contours:
        M = cv2.moments(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        if(radius > 50 and radius <300):
            center = (int(x),int(y))
            radius = int(radius)
            marblesSet.add(Marble(center,radius))
            cv2.circle(newImage,center,radius,(0,255,0),4)
    #showAndWait("marbles contours",newImage)
    # print([str(marble) for marble in marblesSet])
    arrivalLineMarbles = []
    for marble in marblesSet:
        if marble.radius>radiusArrivalObject:
            arrivalLineMarbles.append(marble)

    arrivalLineMarbles.sort(key=getX)
    cv2.line(newImage,arrivalLineMarbles[0].center,arrivalLineMarbles[1].center,(255,0,0),5)
    runningMarbles = set([marble for marble in marblesSet.difference(set(arrivalLineMarbles))])
    arrivalCenter1=arrivalLineMarbles[0].center
    arrivalCenter2=arrivalLineMarbles[1].center
    # print([str(marble) for marble in runningMarbles])
    distances = []
    for marble in runningMarbles : 
        linePoint = getClosestPointOnLine(arrivalCenter1,arrivalCenter2,marble.center)
        if(getSideOfLine(arrivalCenter1,arrivalCenter2,marble.center)):
            cv2.line(newImage,marble.center,linePoint,(0,255,255),3)
        else :
            cv2.line(newImage,marble.center,linePoint,(0,0,255),3)
        marble.distance = math.sqrt((marble.center[0]-linePoint[0])**2+(marble.center[1]-linePoint[1])**2)
        
    closestMarbleValue = min([marble.distance for marble in runningMarbles ])
    closestMarble = [marble for marble in runningMarbles if marble.distance-closestMarbleValue<1]
    print("Closest Marble is at " + str(closestMarbleValue)+"px and is the marble with "+str(closestMarble[0]))
    return newImage

def shapeDetectionOnVideo(img):
    """ Return an image with shape detected """    
    return shapeDetection(img,100,80)

def shapeDetectionOnImage(imagePath):
    img = cv2.imread(imagePath)
    return shapeDetection(img,127,150)
############################################################
############################################################
##############Caracteristiques & Display####################
############################################################
############################################################

def showAndWait(string, img) :
    """ Simple show for img that insure to fit in screen """

    #get a nice coef
    primaryW=int(img.shape[1])
    primaryH=int(img.shape[0])
    coefW=0.6/(primaryW/WIDTH_SCREEN)
    coefH=0.6/(primaryH/HEIGHT_SCREEN)
    coef=max(coefH,coefW)
    
    scale_percent = int(coef*100) # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #display
    cv2.imshow(string, resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def imgHisto(imagePath):
    """ Display the histogram of the image at a given path """
    img=cv2.imread (imagePath)
    #RGB -> HSV.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Déclaration des couleurs des courbes
    color = ('r','g','b')
    #Déclaration des noms des courbes.
    labels = ('h','s','v')
    #Pour col allant r à b et pour i allant de 0 au nombre de couleurs
    for i,col in enumerate(color):
        #Hist prend la valeur de l'histogramme de hsv sur la canal i.
        hist = cv2.calcHist([hsv],[i],None,[256],[0,256])
        # Plot de hist.
        plt.plot(hist,color = col,label=labels[i])
        plt.xlim([0,256])
    #Affichage.
    plt.show()