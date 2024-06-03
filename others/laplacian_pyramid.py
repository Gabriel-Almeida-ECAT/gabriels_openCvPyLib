import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.signal import sepfir2d

im = cv2.imread('test_images/sit_frog.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
h = [1/16, 4/16, 6/16, 4/16, 1/16] # blur filter
N = 4 # number of pyramid levels

# Gaussian pyramid
gaus_pyr = []
gaus_pyr.append(im) # first pyramid level
print(f'Gaussian pyramid [0]: {im.shape}')
for ind in range(1,N):
	im2 = np.zeros(im.shape)
	for z in range(3):
		im2[:,:,z] = sepfir2d( im[:, :, z], h, h) # blur each color chanel
	im2 = im2[0:im.shape[0]:2, 0:im.shape[1]:2, :] #down-sample
	gaus_pyr.append(np.uint8(im2))
	im = im2
	print(f'Gaussian pyramid [{ind}]: {im2.shape}')

# Laplacian pyramid
lapl_pyr = []
for ind in range(0, N-1):
	L1 = gaus_pyr[ind]
	L2 = gaus_pyr[ind+1]
	L2 = cv2.resize(L2, (0,0), fx=2, fy=2) # up-sample
	
	dif = L1  - L2
	dif = dif - np.min(dif) # scale in [0,1]
	dif = dif / np.max(dif) # for display purposes
	lapl_pyr.append(dif)

lapl_pyr.append(gaus_pyr[N-1])

# display pyramid
fig, ax = plt.subplots(nrows=1, ncols=N, figsize=(15,7), dpi=72, sharex=True, sharey=True)
for ind in range(N-1, -1, -1):
	ax[ind].imshow(lapl_pyr[ind])

plt.show()