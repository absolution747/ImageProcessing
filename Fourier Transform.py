import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('cat.jpg',0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum =20*np.log(1 + np.abs(fshift))



#Diplaying the processed images
f, axarr = plt.subplots(3,2)
axarr[0,1].imshow(img, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
axarr[0,0].imshow(magnitude_spectrum, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
#plt.show()

# Making the Mask

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

mask1 = np.zeros((rows, cols), np.uint8)
mask2 = np.ones((rows, cols), np.uint8)

r = 50
center = [crow,ccol]
x, y = np.ogrid[:rows, :cols]
mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
mask1[mask_area] = 1
mask2[mask_area] = 0

#Applying the mask
fshift1 = np.multiply(fshift,mask1)
fshift2 = np.multiply(fshift,mask2)
spectrum1 = 20*np.log(1 + np.abs(fshift1))
spectrum2 = 20*np.log(1 + np.abs(fshift2))
fshift1 = np.fft.ifftshift(fshift1)
fshift2 = np.fft.ifftshift(fshift2)
img_back1 = np.abs(np.fft.ifft2(fshift1))*255
img_back2 = np.abs(np.fft.ifft2(fshift2))*255

# Displaying Results
axarr[1,0].imshow(spectrum1, cmap = 'gray')
#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
axarr[1,1].imshow(img_back1, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
axarr[2,0].imshow(spectrum2, cmap = 'gray')
axarr[2,1].imshow(img_back2, cmap = 'gray')
plt.show()

