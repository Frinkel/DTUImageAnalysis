# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:30:15 2023

@author: Joel
"""

from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import numpy as np
#import pydicom as dicom

# Exercise 1
# Directory containing data and images
in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"

# X-ray image
im_name = "metacarpals.png"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

# Exercise 2
print(im_org.shape)

# Exercise 3
print(im_org.dtype)

# Exercise 4
io.imshow(im_org)
plt.title('Metacarpal image')
io.show()

# Exercise 5
io.imshow(im_org, cmap="jet")
plt.title('Metacarpal image (with colormap)')
io.show()

# Exercise 7
io.imshow(im_org, vmin=20, vmax=170)
plt.title('Metacarpal image (with gray level scaling)')
io.show()

min_val = np.min(im_org)
max_val = np.max(im_org)
io.imshow(im_org, vmin=min_val, vmax=max_val)
plt.title('Metacarpal image (with gray level scaling of min max)')
io.show()


# Exercise 8
plt.hist(im_org.ravel(), bins=256)
plt.title('Image histogram')
io.show()

h = plt.hist(im_org.ravel(), bins=256)

bin_no = 100
count = h[0][bin_no]
print(f"There are {count} pixel values in bin {bin_no}")

bin_left = h[1][bin_no]
bin_right = h[1][bin_no + 1]
print(f"Bin edges: {bin_left} to {bin_right}")

# y, x, _ = plt.hist(im_org.ravel(), bins=256)

# Exercise 9

h0 = h[0].argmax()
print(f"Common range is {h0}")


# Exercise 10
r = 100
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

r = 110
c = 90
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")


# Exercise 11
im_org[:30] = 0
io.imshow(im_org)
io.show()

mask = im_org > 150
io.imshow(mask)
io.show()


# Exercise 13
im_org[mask] = 255
io.imshow(im_org)
io.show()


# Exercise 14

im_name = "ardeche.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)
print(im_org.dtype)

r = 100
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# Exercise 16
num_rows = im_org.shape[0]
half_rows = int(num_rows // 2)
im_org[:half_rows] = [0, 255, 0]
io.imshow(im_org)
io.show()


# Exercise 17
in_dir = "D:/GitHub/DTUImageAnalysis/exercises-solved/"
im_name = "reyna_img.jpg"

# Read the image.
# Here the directory and the image name is concatenated
# by "+" to give the full path to the image.
im_org = io.imread(in_dir + im_name)

print(im_org.shape)
print(im_org.dtype)

r = 100
c = 50
im_val = im_org[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

# Exercise 18
image_rescaled = rescale(im_org, 0.25, anti_aliasing=True,
                         channel_axis=2)

im_val = image_rescaled[r, c]
print(f"The pixel value at (r,c) = ({r}, {c}) is: {im_val}")

io.imshow(image_rescaled)
io.show()


# Exercise 19
image_resized = resize(im_org, (im_org.shape[0],
                       400),
                       anti_aliasing=True)
io.imshow(image_rescaled)
io.show()

# Gray-scale image
im_gray = color.rgb2gray(im_org)
im_byte = img_as_ubyte(im_gray)


plt.hist(im_byte.ravel(), bins=256)
plt.title('Image histogram')
io.show()

h = plt.hist(im_org.ravel(), bins=256)




# Exercise 22
in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"
im_name = "DTUSign1.jpg"
im_org = io.imread(in_dir + im_name)

r_comp = im_org[:, :, 0]
io.imshow(r_comp)
plt.title('DTU sign image (Red)')
io.show()

g_comp = im_org[:, :, 1]
io.imshow(g_comp)
plt.title('DTU sign image (Green)')
io.show()

b_comp = im_org[:, :, 2]
io.imshow(b_comp)
plt.title('DTU sign image (Blue)')
io.show()




# Exercise 24
in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"
im_name = "DTUSign1.jpg"
im_org = io.imread(in_dir + im_name)

im_org[500:1000, 800:1500, :] = 0

io.imshow(im_org)
plt.title("Black rectangle")
io.show();

io.imsave("DTUSign1-marked.png", im_org)


# Exercise 27
in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"
im_name = "metacarpals.png"
im_org = io.imread(in_dir + im_name)


im_col = color.gray2rgb(im_org)

mask = im_org > 150

im_col[mask] = (0, 0, 255)

io.imshow(im_col)
io.show()


# Advanced Image Visualization
in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"
im_name = "metacarpals.png"
im_org = io.imread(in_dir + im_name)

p = profile_line(im_org, (342, 77), (320, 160))
plt.plot(p)
plt.ylabel('Intensity')
plt.xlabel('Distance along line')
plt.show()


in_dir = "D:/GitHub/DTUImageAnalysis/exercises/ex1-IntroductionToImageAnalysis/data/"
im_name = "road.png"
im_org = io.imread(in_dir + im_name)
im_gray = color.rgb2gray(im_org)
ll = 200
im_crop = im_gray[40:40 + ll, 150:150 + ll]
xx, yy = np.mgrid[0:im_crop.shape[0], 0:im_crop.shape[1]]
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xx, yy, im_crop, rstride=1, cstride=1, cmap=plt.cm.jet,
                       linewidth=0)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
