#!/usr/bin/env python
""" Provides every tools used by main file
The following functions allow to get and draw shapes on image to detect marbles.
"""
__author__="Adrien Paysant, Joris Monnet"

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics
from shapely.geometry import LineString
from shapely.geometry import Point
import math

WIDTH_SCREEN = GetSystemMetrics(0)
HEIGHT_SCREEN = GetSystemMetrics(1)

#############################################################################################################
# Class Marble
#############################################################################################################
class Marble:
    """Class used as a container to get center, radius and distance of marbles"""
    def __init__(self,center,radius):
        self.center=center
        self.radius=radius
        self.distance=-1
    
    def __str__(self):
        return f"center = {self.center} and radius = {self.radius}"  
  
#############################################################################################################
# Mathematic tools
#############################################################################################################
# from https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
def getIntersectionLineCircle(center,radius,x1,y1,x2,y2):
    """return the intersection between a circle(marble of arrival) and a line(arrival line)"""
    p = Point(center[0],center[1])
    c = p.buffer(radius).boundary
    l = LineString([(x1,y1), (x2, y2)])
    return  c.intersection(l)

#https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def getClosestPointOnLine(marble1,marble2,marbleRunning):
    """return the closest point on the line frome the center of the marble"""
    x1, y1 = marble1.center
    x2, y2 = marble2.center
    x3, y3 = marbleRunning.center
    dx, dy = x2-x1, y2-y1
    coef = dx*dx + dy*dy
    a = (dy*(y3-y1)+dx*(x3-x1))/coef
    x = int(x1+a*dx)
    y = int(y1+a*dy)
    
    marble1Intersection = getIntersectionLineCircle(marble1.center,marble1.radius,x1,y1,x2,y2)
    marble2Intersection = getIntersectionLineCircle(marble2.center,marble2.radius,x1,y1,x2,y2)
    marble1MinPoint = marble1Intersection.coords[0]
    marble2MinPoint = marble2Intersection.coords[0]
    x1 = int(marble1MinPoint[0])
    x2 = int(marble2MinPoint[0])
    y1 = int(marble1MinPoint[1])
    y2 = int(marble2MinPoint[1])

    xMin = min(x1,x2)
    xMax = max(x1,x2)
    yMin = min(y1,y2)
    yMax = max(y1,y2)
    xFinal = x
    yFinal = y
    #Check x minimum and maximum
    if(xMin<=x):
        if(x>xMax):
            xFinal=xMax
    else :
        xFinal=xMin

    #Check y minimum and maximum
    if(yMin<=y):
        if(y>yMax):
            yFinal=yMax
    else :
        yFinal=yMin
    return xFinal,yFinal

def getSideOfLine(p1,p2,p3):
    """return if the marble crossed the arrival line"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    if (x2-x1)*(y2-y3) - (y2-y1)*(x2-x3) > 0:
        return False
    print ('A marble crossed the line')
    return True

#############################################################################################################
# Image Tools
#############################################################################################################
def getMask(grayImage,lowerValueRange,upperValueRange,ksize):
    """create a mask from a grayimage with a blur and an inRange()"""
    mask = cv2.GaussianBlur(grayImage, (ksize, ksize), 0)
    mask = cv2.inRange(mask, lowerValueRange, upperValueRange)
    return mask

def erodeDilate(grayImage,mask, erodeIterations,dilateIterations):
    """erode and dilate image"""
    res = cv2.bitwise_and(grayImage, grayImage, mask=mask)
    res = cv2.erode(res, None, iterations=erodeIterations)
    res = cv2.dilate(res, None, iterations=dilateIterations)
    return res

def getY(elem):
    """return y value of center of a marble to sort it"""
    return elem.center[1]

#Main function of the file
def shapeDetection(img,lowerValueRange,upperValueRange,radiusArrivalObject,radiusMin,radiusMax,erodeIterations,dilateIterations,ksize,debug):
    """find and draw contours and shapes
    img: image already read
    lowerValueRange: lower value of color for getMask
    upperValueRange: upper value of color
    radiusArrivalObject: size of radius minimum for arrival object
    radiusMin: minimum radius to take an area detected as a marble
    radiusMax : maximum radius of a marble
    erodeIterations: number of erode iterations
    dilateiterations: number of dilate iterations
    ksize: size of kernel for blur
    debug: True if we want to display closest marble"""
    grayImage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = getMask(grayImage,lowerValueRange,upperValueRange,ksize)
    result = erodeDilate(grayImage,mask,erodeIterations,dilateIterations)

    newImage = img
    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(newImage, contours, -1, (255, 0, 0), 5)

    marblesSet = set()
    for cnt in contours:
        M = cv2.moments(cnt)
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        if(radius > radiusMin and radius <radiusMax):
            center = (int(x),int(y))
            radius = int(radius)
            marblesSet.add(Marble(center,radius))
            cv2.circle(newImage,center,radius,(0,255,0),2)

    # showAndWait("marbles contours",newImage)
    arrivalLineMarbles = []
    for marble in marblesSet:
        if marble.radius>radiusArrivalObject:
            arrivalLineMarbles.append(marble)

    arrivalLineMarbles.sort(key=getY,reverse=True)
    cv2.line(newImage,arrivalLineMarbles[0].center,arrivalLineMarbles[1].center,(255,0,0),5)
    runningMarbles = set([marble for marble in marblesSet.difference(set(arrivalLineMarbles))])
    arrivalCenter1=arrivalLineMarbles[0].center
    arrivalCenter2=arrivalLineMarbles[1].center

    distances = []
    for marble in runningMarbles : 
        linePoint = getClosestPointOnLine(arrivalLineMarbles[0],arrivalLineMarbles[1],marble)
        if(getSideOfLine(arrivalCenter1,arrivalCenter2,marble.center)):
            cv2.line(newImage,marble.center,linePoint,(0,255,255),2)
        else :
            cv2.line(newImage,marble.center,linePoint,(0,0,255),2)
        marble.distance = math.sqrt((marble.center[0]-linePoint[0])**2+(marble.center[1]-linePoint[1])**2)
    if len(runningMarbles)> 0:
        closestMarbleValue = min([int(marble.distance) for marble in runningMarbles ])
        closestMarble = [marble for marble in runningMarbles if marble.distance-closestMarbleValue<1]
        if(debug):
            print("Closest Marble is at " + str(closestMarbleValue)+"px and is the marble with "+str(closestMarble[0]))
    return newImage

def shapeDetectionOnOldVideo(img):
    """get shapedetection for data.mp4"""
    return shapeDetection(img,100,255,80,50,300,3,1,11,True)

def shapeDetectionOnVideo(img):
    """get shapedetection for data2.mp4"""
    return shapeDetection(img,70,255,80,10,300,10,9,7,False)

def shapeDetectionOnImage(imagePath):
    """get shapedetection for [2-5].jpg"""
    img = cv2.imread(imagePath)
    return shapeDetection(img,90,255,100,30,300,0,7,7,False)

def shapeDetectionOnOldImage(imagePath):
    """get shapedetection for 0.jpg and 1.jpg"""
    img = cv2.imread(imagePath)
    return shapeDetection(img,127,255,150,50,300,3,1,11,False)

#############################################################################################################
# Display image
#############################################################################################################
def showAndWait(string, img) :
    """ Simple show for image that insure to fit in screen """
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