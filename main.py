import ImageTools 
import os
import cv2


if __name__ == "__main__":
    print('Main Launching')


    path='./source_images/'
    file="0.jpg"
    ImageTools.shapeDetection(os.path.join(path, file))


    cv2.waitKey(0)