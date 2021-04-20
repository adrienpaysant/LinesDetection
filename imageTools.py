import os
import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from win32api import GetSystemMetrics

WIDTH_SCREEN= GetSystemMetrics(0)
HEIGHT_SCREEN= GetSystemMetrics(1)

############################################################
############################################################
###################Basics Transforms########################
############################################################
############################################################

def hsvSpace(imagePath):
    img=cv2.imread (imagePath)
    return  cv2.cvtColor(img, cv2.COLOR_BGR2HSV)



    
############################################################
############################################################
####################Shape Detection#########################
############################################################
############################################################

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

    return img

def areaDetection(imgPath):
    img=cv2.imread (imgPath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v= cv2.split(hsv)
    ret_h, th_h = cv2.threshold(h,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_s, th_s = cv2.threshold(s,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #Fusion th_h et th_s
    th=cv2.bitwise_or(th_h,th_s)
    #Ajouts de bord à l'image
    bordersize=10
    th=cv2.copyMakeBorder(th, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    #Remplissage des contours
    im_floodfill = th.copy()
    h, w = th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    th = th | im_floodfill_inv
    #Enlèvement des bord de l'image
    th=th[bordersize: len(th)-bordersize,bordersize: len(th[0])-bordersize]
    resultat=cv2.bitwise_and(img,img,mask=th)
    cv2.imwrite("output/im_floodfill.png",im_floodfill)
    cv2.imwrite("output/th.png",th)
    cv2.imwrite("output/resultat.png",resultat)
    contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for i in range (0, len(contours)) :
        mask_BB_i = np.zeros((len(th),len(th[0])), np.uint8)
        x,y,w,h = cv2.boundingRect(contours[i])
        cv2.drawContours(mask_BB_i, contours, i, (255,255,255), -1)
        BB_i=cv2.bitwise_and(img,img,mask=mask_BB_i)
        if h >15 and w>15 :
            BB_i=BB_i[y:y+h,x:x+w]
            cv2.imwrite("output/BB_"+str(i)+".png",BB_i)

############################################################
############################################################
####################Edges Detection#########################
############################################################
############################################################

def edgeDetectAndShowCanny(imagePath):
    """ edge detection with Canny method for image at given path
        and display the edges image & original
      """
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'hsv')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def edgeDetectAndShowHough(imagePath):
    """ edge detection with Hough method for image at given path
        and display the edges image 
      """
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


def imgCaract(imagePath):
    """ print few infos of the image at the given path"""
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