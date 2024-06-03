import matplotlib.pyplot as plt
import numpy as np
import cv2

from numpy import fft

f = cv2.imread('test_images/sit_frog.jpg', 0)
F = fft.fftshift(fft.fft2(f)) # shift the origin to the middle so to clearly see the magnitude of the fourier transform
Fmag1 = np.abs(F)
# That way there are artifical artefacts because the fft expects a periodic signal

#f = cv2.imread('test_images/sit_frog.jpg', 0)
[ydim,xdim] = f.shape
win = np.outer(np.hanning(ydim), np.hanning(xdim))
win = win / np.mean(win) # make unit-mean
F = fft.fftshift(fft.fft2(f*win)) #force periodicity in the input image
Fmag2 = np.abs(F)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.title('Fourier magnitude')
plt.imshow(Fmag1, cmap='gray')

plt.subplot(2,2,2)
plt.title('log(Fourier magnitude)')
plt.imshow(np.log(Fmag1), cmap='gray')

plt.subplot(2,2,3)
plt.title('windowed image')
#plt.imshow(Fmag, cmap='gray')
plt.imshow(f*win, cmap='gray')

plt.subplot(2,2,4)
plt.title('log(Fourier magnitude)')
plt.imshow(np.log(Fmag2), cmap='gray')
plt.show()