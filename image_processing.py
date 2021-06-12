# image_processing.py
# image processing with several techniques
# author: Ryusei Sashida
# created: 26 March 2021

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.filters import gaussian, laplace
from skimage.segmentation import slic
from skimage import feature
from scipy.ndimage import uniform_filter
from skimage.transform import probabilistic_hough_line
from skimage import io, segmentation, color
from skimage.util import img_as_ubyte, img_as_uint

#-----part1-----
print("---part1---")
#load image
ave_img=io.imread('data/image_data/avengers_imdb.jpg')
print("image shape:", ave_img.shape)
#rgb to gary
gray=img_as_ubyte(rgb2gray(ave_img))
io.imsave('outputs/gray.png', gray)

#compute threshhold and convert the grayscale image to binary image
thresh=threshold_otsu(gray)
#check each pixel > threshold
binary=gray > thresh
io.imsave('outputs/binary.png', img_as_uint(binary))

#-----part2-----
#load image
bush_img=io.imread('data/image_data/bush_house_wikipedia.jpg')
#apply nouse
noise_img=img_as_ubyte(random_noise(bush_img, mode='gaussian', var=0.1))
io.imsave('outputs/noise.png', noise_img)
#apply guassian filter
gaus_img=img_as_ubyte(gaussian(noise_img, sigma=1, multichannel=True))
io.imsave('outputs/gaus.png', gaus_img)
#apply smoothing filter
smoot_img=img_as_ubyte(uniform_filter(noise_img, size=(9, 9, 1)))
io.imsave('outputs/smooth.png', smoot_img)

#-----part3-----
#load image
gov_img=io.imread('data/image_data/forestry_commission_gov_uk.jpg')
#k-means(k=5) segementation(color space) 
label=segmentation.slic(gov_img, n_segments=5, start_label=1)
#create segmented image
out=color.label2rgb(label, gov_img, kind = 'avg', bg_label=0).astype(dtype='uint8')
io.imsave('outputs/segmented.png', out)


#-----part4-----
#load image
rolland_img=io.imread('data/image_data/rolland_garros_tv5monde.jpg')
#RGB to Grayscal
gray_rolland_img=rgb2gray(rolland_img)
#Canny edge detection
edges_img=feature.canny(gray_rolland_img)
io.imsave('outputs/edges.png', img_as_uint(edges_img))

#Apply probabilistic hough transform
lines=probabilistic_hough_line(edges_img, threshold=10, line_length=100, line_gap=3)
fig, ax=plt.subplots(1, 1, figsize=(5, 5), sharex=True, sharey=True)
for line in lines:
    p0, p1=line
    ax.plot((p0[0], p1[0]), (p0[1], p1[1]),lw=3)
ax.set_xlim((0, gray_rolland_img.shape[1]))
ax.set_ylim((gray_rolland_img.shape[0], 0))
ax.set_title('Probabilistic Hough Transformation')
fig.savefig("outputs/hough.png")
print("all outputs are saved in outputs folder for all parts")
