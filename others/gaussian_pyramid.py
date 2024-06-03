import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.signal import sepfir2d

'''im = cv2.imread('test_images/sit_frog.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)'''
im = plt.imread('test_images/sit_frog.jpg')
h = [1/16, 4/16, 6/16, 4/16, 1/16] # blur filter
#this is a unit sum filter -> the values add to 1 so to not change the brightness
N = 3 # number of pyramid levels

p = []
p.append(im) # first pyramid level
print(im.shape)
for k in range(1,N):
	im2 = np.zeros(im.shape)
	for z in range(3):
		im2[:,:,z] = sepfir2d( im[:, :, z], h, h) # blur each color chanel
	im2 = im2[0:im.shape[0]:2, 0:im.shape[1]:2, :] #down-sample
	p.append(np.uint8(im2))
	im = im2
	print(im2.shape)

#display pyramid
'''for ind, img in enumerate(p, start=1):
	plt.subplot(1, N, ind)
	plt.imshow(img)
	plt.axis('off')'''

'''index = 0
print(p[index])
print(p[index].shape)
print(f'max: {np.max(p[index])} - min: {np.min(p[index])}')
plt.imshow(p[index])
plt.show()'''

fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(15,7), dpi=72, sharex=True, sharey=True)
for ind in range(N-1, -1, -1):
	ax[ind].imshow(p[ind])

plt.show()