import cv2 
import numpy as np
  
cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of red color in HSV
    lower_red = np.array([0,20,20])
    upper_red = np.array([0,255,255])

    # Threshold the HSV image to get only red colors
    segment = cv2.inRange(hsv, lower_red, upper_red)
    segment_closing = cv2.morphologyEx(segment, cv2.MORPH_CLOSE, kernel,
                                       iterations = 3)

    # Bitwise-AND mask and original image
    red = cv2.bitwise_and(frame,frame, mask= segment_closing)

    cv2.imshow('frame',frame)
    cv2.imshow('segement',segment)
    cv2.imshow('mask',segment_closing)
    cv2.imshow('red',red)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

