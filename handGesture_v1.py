#!/usr/bin/python2.7

#Import necessary packages
import cv2
import numpy
import math

capture = cv2.VideoCapture(0)

keyPressed = -1

# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([0, 138, 67],numpy.uint8)
max_YCrCb = numpy.array([255,173,133],numpy.uint8)

# Small rectangular kernel for the opening
kernel = numpy.ones((7,7),numpy.uint8)

while(keyPressed != 27):

    _, frame = capture.read()

    #Flip the image to get a mirror effect
    frame = cv2.flip(frame,1)

    #Define ROI
    roi = frame[50:300,300:550]
    cv2.rectangle(frame,(300,50),(550,300),(255,0,0),2)


    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    #skinRegion = cv2.GaussianBlur(skinRegion, (3,3), 0)
    #Morphologic Opening to remove the noise
    skinRegion = cv2.morphologyEx(skinRegion, cv2.MORPH_OPEN, kernel)

    # Do contour detection on skin region
    _, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contour on the source image
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        #if the contour have a minimum size (to remove the noise)
        if area > 1000:
            cnt = contours[i]

            hull = cv2.convexHull(cnt)

            #Drawing contour and convex hull
            cv2.drawContours(roi, [cnt], 0, (0,0,255), 0)
            cv2.drawContours(roi, [hull], 0, (0,255,0), 0)

            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt,hull)

            count_defects = 0

            #http://vipul.xyz/2015/03/gesture-recognition-using-opencv-python.html
            if hasattr(defects, 'shape'):
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]

                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    # find length of all sides of triangle
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

                    # apply cosine rule here
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

                    # ignore angles > 90 and highlight rest with red dots
                    if angle <= 90:
                        count_defects += 1

                #Display number of fingers
                if count_defects == 1:
                    cv2.putText(frame,"2", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
                elif count_defects == 2:
                    cv2.putText(frame,"3", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
                elif count_defects == 3:
                    cv2.putText(frame,"4", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
                elif count_defects == 4:
                    cv2.putText(frame,"5", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
                elif count_defects == 0:
                    cv2.putText(frame,"1", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)


    cv2.imshow('Camera Output',frame)
    cv2.imshow('',skinRegion)

    keyPressed = cv2.waitKey(1)

capture.release()
