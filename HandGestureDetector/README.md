# Hand Gesture Detector

## Intro

 In order to use the image processing's concepts I learned during the first year of my degree I wanted to do a small and fun program. My first idea was to create a hand tracking program similar to something like [that](https://vimeo.com/18103562) where you detect the position of the hand and create an animation. Even without doing a complex animation like this I'd like to be able to do simple drawing with one hand while the other select the color.

After some research on the subject and some test I encountered two main problems:

- In his project Andrew Berg uses a kinect, which is equipped with a depth sensor (more about it [here](https://www.youtube.com/watch?v=uq9SEJxZiUg) so instead of processing straight from the video of the webcam like I wanted to do, he already has a notion of depth and can separate the important information from the background. This mean I had to find a way to subtract the background of my images, openCV has functions to do it (BackgroundSubtractorMOG2,...) but the object need to stay in movement and this wasn't always the case in my tests.
- I'm developing in my bed room where the wall are covered with posters and there is a big window so the light change a lot throughout the day. I originally wanted to develop my hand tracking program to be robust enough so it can work with every background and light, I realized that this was a little too ambitious and I stepped down my project to this version: a hand gesture recognition program working with a clean background and my skin's color.

This project was made in python using the openCV library.

## Processing explanation

### Vidéo capture



The first step is to get the images I'm going to process. I am using my camera and I'll get and treat each frame individually. To do that I used two openCV function:

- **cv2.VideoCapture(0)** create a VideoCapture object needed to read the frames later. Its arguments can be the camera index (0 for the default) or the name of a video file.
- **capture.read()** applied on the VideoCapture object, it'll return a boolean (true if the frame was read correctly) and the next frame of the video.

By using it in a loop I can access each frame of the video and treat them in real time. 

```
#Import necessary packages
import cv2
import numpy
import math

#Create the VideoCaptureObject
capture = cv2.VideoCapture(0)

#Loop until the esc key is pressed
keyPressed = -1
while(keyPressed != 27):

  #Get the next frame
  _, frame = capture.read()

  #Our operations goes here

  #Display the frame
  cv2.imshow('Camera Output',frame)

  keyPressed = cv2.waitKey(1)

#At the end release the capture
capture.release()
```

### Hand extraction

 This was the hardest part and the one that need improvement the most. As I said in the introduction my first idea was to use a background subtraction algorithm to make the program robust and usable with all kinds of backgrounds, but I couldn't find a solution that works with a stationary object (and with only one camera).

To make it easier I defined a ROI, a smaller portion of my image which will contain the hand and nothing else. 

```
#Define ROI and  draw it
roi = frame[50:300,300:550]
cv2.rectangle(frame,(300,50),(550,300),(255,0,0),2)
```

![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_1.jpg)

Now that we have a clean ROI the goal is to create a binary mask of the hand. I went with a segmentation based on the skin color using the inRange openCV function and the YCbCr color space. After some research it seems that there isn't a best color space when it comes to skin detection as long as you use a good detector (but I might try the HSV color space to see if I get better results). You can find working range on the internet and just tweak them a little. Every pixel whose values are between the range in the YCbCr image will be white in our mask. 

```
# Constants for finding range of skin color in YCrCb
min_YCrCb = numpy.array([0, 138, 67],numpy.uint8)
max_YCrCb = numpy.array([255,173,133],numpy.uint8)

#Convert image to YCrCb
imageYCrCb = cv2.cvtColor(roi,cv2.COLOR_BGR2YCR_CB)

#Find region with skin tone in YCrCb image
skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
```
![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_2.jpg)
![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_3.jpg)

### Contour and convex hull

We are using an openCV function to find the contour of our hand using the binary image created and draw them. 

```
#Do contour detection on skin region
_, contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Draw the contour on the source image
for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    #if the contour have a minimum size (to remove the noise)
    if area > 1000:
        cnt = contours[i]

        #Drawing contour
        cv2.drawContours(roi, [cnt], 0, (0,0,255), 0)
```

With this contour we can use the openCV function cv2.convexHull to find the convex hull of the hand. The convex hull of a set of N points is the smallest perimeter fence enclosing the points. 

![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_5.png)

```
#Find the convel hull of the hand returnPoints
hull = cv2.convexHull(cnt)

#Drawing convex hull
cv2.drawContours(roi, [hull], 0, (0,255,0), 0)
```

![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_4.jpg)

### Convexity defect and finger counting

 We now have everything we need to count the number of raised fingers. The idea is to count the number of defect point. To detect the convexity defect we are using the function **cv2.convexityDefects(cnt,hull)**

To put it simply the convexity defect region is the areas that do not belong to the object but are located inside of its convex hull. It can be resume to a starting point P0, an ending point p1 and the defect point p2. A common approach and the one I went with is to calculate the angle alpha of the gap between two finger. If it is sufficiently small we consider the finger to be raised. 

![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_6.png)
![alt text](https://raw.githubusercontent.com/Aelly/Image-processing/master/ImageDoc/HandGesture_7.png)

```
for i in range(defects.shape[0]):
  s,e,f,d = defects[i,0]

  start = tuple(cnt[s][0])
  end = tuple(cnt[e][0])
  far = tuple(cnt[f][0])

  #Find length of all sides of triangle
  a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
  b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
  c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)

  #Apply cosine rule here
  angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57

  #Ignore angles > 90 and highlight rest with red dots
  if angle <= 90:
    count_defects += 1

  #Display number of fingers
  if count_defects == 1:
    cv2.putText(frame,"2", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
  elif count_defects == 2:
    cv2.putText(frame,"3", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 10)
  elif count_defects == 3:
      ...
```

The code for this part was found [here](https://github.com/vipul-sharma20/gesture-opencv)

### Conclusion

This is a simple way to do a hand gesture recognition program but now that i understand this i'll find ways to improve it:

- Change the finger counting part. By using the angle of the gap between the finger we can't differenciate one or zero finger
- Improve the hand detection to smooth the contour and be able to use it with a less clean background
