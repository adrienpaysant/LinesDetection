import ImageTools 
import os
import cv2


if __name__ == "__main__":
    print('Main Launching')


    path='./source_images/'
    file="0.jpg"
    imgPath=os.path.join(path, file)
    #ImageTools.areaDetection(os.path.join(path, file))

    #ImageTools.edgeDetectAndShowCanny(imgPath)
    #ImageTools.edgeDetectAndShowHough(imgPath)
    dataHSV=ImageTools.hsvSpace(imgPath)
    ImageTools.imgCaract(imgPath)
    ImageTools.showAndWait("Histo",ImageTools.imgHisto(imgPath))
    ImageTools.showAndWait("HSV Space",dataHSV)

    cv2.waitKey(0)

    print('Main End')