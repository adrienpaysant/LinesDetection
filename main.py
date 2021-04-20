import imageTools 
import os
import cv2


if __name__ == "__main__":
    print('Main Launching')


    path='./source_images/'
    file="0.jpg"
    imgPath=os.path.join(path, file)
    #imageTools.areaDetection(os.path.join(path, file))

    #ImageTools.edgeDetectAndShowCanny(imgPath)
    #ImageTools.edgeDetectAndShowHough(imgPath)
    dataHSV=imageTools.hsvSpace(imgPath)
    #imageTools.imgCaract(imgPath)
    imageTools.imgHisto(imgPath)
    imageTools.showAndWait("HSV Space",dataHSV)

    cv2.waitKey(0)
    print('Main End')