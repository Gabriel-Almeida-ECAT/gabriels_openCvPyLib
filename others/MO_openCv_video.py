import cv2
import numpy as np
import copy
import os
import time

from gabriels_openCvPyLib import openCvLib


filter_opt = '3'
desired_size = (600,600)

if __name__ == '__main__':
    source = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    #source = cv2.VideoCapture('video_blue.mp4')

    if (source.isOpened()== False): 
        print("Error opening video stream or file")

    win_name = 'Camera'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    frame_width = int(source.get(3))
    frame_height = int(source.get(4))

    frame_rate = source.get(cv2.CAP_PROP_FPS)
    print(f"Webcan frame_rate = {frame_rate}")


    while cv2.waitKey(1) != 27: #scape
        has_frame, crtFrame = source.read()

        if not has_frame:
            break

        if filter_opt == '1':
            crtFrame = cv2.resize(crtFrame, desired_size)

            cv2.imshow(win_name, crtFrame)

        elif filter_opt == '2':
            crtFrame = cv2.resize(crtFrame, desired_size)

            cv2.imshow(win_name, crtFrame)


        elif filter_opt == '3':
            crtFrame = cv2.resize(crtFrame, desired_size)

            cv2.imshow(win_name, crtFrame)

        #time.sleep(1/frame_rate) #run on the original video speed

    source.release()
    cv2.destroyWindow(win_name)