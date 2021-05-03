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

    for i in range(2):
        imageTools.showAndWait("basic shape detect",imageTools.shapeDetectionOnImage(f"./source_images/{i}.jpg"))

    video = cv.VideoCapture("./source_videos/data.mp4")
    newVideo = cv.VideoWriter("newVideo.avi",cv.CAP_OPENCV_MJPEG,cv.VideoWriter_fourcc('M','J','P','G'),10,(1920,1080),True)
    numberOfFrame = 0
    while(True):
        ret,frame = video.read()

        if ret:
            img = imageTools.shapeDetectionOnVideo(frame)
            newVideo.write(img)
            numberOfFrame += 1
        else:
            break
    print(f"Video has {numberOfFrame} frames")
    # Release all space and windows once done
    video.release()
    newVideo.release()
    cv.waitKey(0)
    print('Main End')