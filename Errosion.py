import cv2
import numpy as np

img = cv2.imread('cat.jpg',0)
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)
dilation = cv2.dilate(img, kernel, iterations=1)

Gradient = dilation - erosion
dilation = np.divide(dilation,255)
erosion  = np.divide(erosion,255)

#cv2.imshow('Input', img)
#cv2.imshow('Erosion', erosion)
#cv2.imshow('Dilation', dilation)
cv2.imshow('Gradient', Gradient)

cv2.waitKey(0)
