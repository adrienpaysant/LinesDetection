import imageTools 
import os
import cv2 as cv
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print('Main Launching')


    path='./source_images/'
    file="0.jpg"
    imgPath=os.path.join(path, file)

    # for i in range(2):
    #     imageTools.showAndWait("basic shape detect",imageTools.shapeDetectionOnImage(f"./source_images/{i}.jpg"))

    video = cv.VideoCapture("./source_videos/data.mp4")

    currentFrame = 0
    while(True):

        ret,frame = video.read()
        print("read")
    
        if ret:
            print("ret")
            imageTools.showAndWait("basic shape detect",imageTools.shapeDetectionOnVideo(frame))
    
            # increasing counter so that it will
            # show how many frames are created
            currentFrame += 1
        else:
            break
    
    # Release all space and windows once done
    video.release()
    cv.waitKey(0)
    print('Main End')