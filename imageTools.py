import os
import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt

def edgeDetectAndShow(imagePath):
    img = cv2.imread(imagePath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'hsv')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()



def showAndWait(img, string) :
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