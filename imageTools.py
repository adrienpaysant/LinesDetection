import os
import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics
import sys
import math

sys.setrecursionlimit(10000)


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

def shapeDetectionOnImage(imagePath):
    """ Return an image with shape detected
        Note : this is working very badly
        """
    img = cv2.imread(imagePath)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mask = cv2.GaussianBlur(hsv, (11, 11), 0)
    mask = cv2.inRange(mask, 127, 255)
    
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res = cv2.erode(res, None, iterations=3)
    res = cv2.dilate(res, None, iterations=1)
    
    newImage = 255* np.ones((img.shape),dtype=np.uint8)
    #newImage = img
    contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImage, contours, -1, (255, 0, 0), 5)
    showAndWait("test",newImage)
    marblesSet = set()
    for cnt in contours:
        M = cv2.moments(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)

        if(radius > 40 and radius <300):
            center = (int(x),int(y))
            radius = int(radius)
            marblesSet.add(Marble(center,radius))
            cv2.circle(newImage,center,radius,(0,255,0),4)
            print("found")
    #showAndWait("marbles contours",newImage)
    # print([str(marble) for marble in marblesSet])
    arrivalLineMarbles = []
    for marble in marblesSet:
        if marble.radius>150:
            arrivalLineMarbles.append(marble)

    cv2.line(newImage,arrivalLineMarbles[0].center,arrivalLineMarbles[1].center,(255,0,0),5)
    runningMarbles = set([marble for marble in marblesSet.difference(set(arrivalLineMarbles))])
    arrivalCenter1=arrivalLineMarbles[0].center
    arrivalCenter2=arrivalLineMarbles[1].center
    # print([str(marble) for marble in runningMarbles])
    distances = []
    for marble in runningMarbles : 
        linePoint = getClosestPointOnLine(arrivalCenter1,arrivalCenter2,marble.center)
        marble.distance = math.sqrt((marble.center[0]-linePoint[0])**2+(marble.center[1]-linePoint[1])**2)
        cv2.line(newImage,marble.center,linePoint,(0,0,255),3)

    print("closest Marble is at " + str(min([marble.distance for marble in runningMarbles ])))
    return newImage          

#https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def getClosestPointOnLine(p1,p2,p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx, dy = x2-x1, y2-y1
    coef = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/coef
    return int(x1+a*dx), int(y1+a*dy)

def shapeDetectionOnVideo(img):
    """ Return an image with shape detected
        Note : this is working very badly
        """    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    mask = cv2.GaussianBlur(hsv, (11, 11), 0)
    mask = cv2.inRange(mask, 100, 255)
    
    res = cv2.bitwise_and(hsv, hsv, mask=mask)
    res = cv2.erode(res, None, iterations=3)
    res = cv2.dilate(res, None, iterations=1)
    
    newImage = 255* np.ones((img.shape),dtype=np.uint8)
    #newImage = img
    contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            print("found")
    #showAndWait("marbles contours",newImage)
    # print([str(marble) for marble in marblesSet])
    arrivalLineMarbles = []
    for marble in marblesSet:
        if marble.radius>80:
            arrivalLineMarbles.append(marble)

    cv2.line(newImage,arrivalLineMarbles[0].center,arrivalLineMarbles[1].center,(255,0,0),5)
    runningMarbles = set([marble for marble in marblesSet.difference(set(arrivalLineMarbles))])
    arrivalCenter1=arrivalLineMarbles[0].center
    arrivalCenter2=arrivalLineMarbles[1].center
    # print([str(marble) for marble in runningMarbles])
    distances = []
    for marble in runningMarbles : 
        linePoint = getClosestPointOnLine(arrivalCenter1,arrivalCenter2,marble.center)
        marble.distance = math.sqrt((marble.center[0]-linePoint[0])**2+(marble.center[1]-linePoint[1])**2)
        cv2.line(newImage,marble.center,linePoint,(0,0,255),3)

    print("closest Marble is at " + str(min([marble.distance for marble in runningMarbles ])))
    return newImage        
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