
import cv2
import numpy as np
import matplotlib.pyplot as plt

Image_name = 'Tipblight 6.jpg'

# opencv loads the image in BGR, convert it to RGB
img = cv2.cvtColor(cv2.imread(Image_name),
                   cv2.COLOR_BGR2RGB)
lower_green = np.array([0, 120, 0], dtype=np.uint8)
upper_green = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(img, lower_green, upper_green)  # could also use threshold
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))  # "erase" the small white points in the resulting mask
mask = cv2.bitwise_not(mask)  # invert mask

# load background (could be an image too)
bk = np.full(img.shape, 200, dtype=np.uint8)  # white bk

# get masked foreground
fg_masked = cv2.bitwise_and(img, img, mask=mask)

# get masked background, mask must be inverted 
mask = cv2.bitwise_not(mask)
bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

# combine masked foreground and masked background 
final = cv2.bitwise_or(fg_masked, bk_masked)
mask = cv2.bitwise_not(mask)  # revert mask to original


import scipy.misc
scipy.misc.imsave('filter_orig_im.jpg', final)
scipy.misc.imsave('filter_orig_im_mask.jpg', mask)



frame =  cv2.imread(Image_name)

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

upper_green = np.array([75,255,255])
lower_green = np.array([20,100,50])

mask = cv2.inRange(hsv, lower_green, upper_green)
res = cv2.bitwise_and(frame,frame, mask= mask)

kernel = np.ones((15,15),np.float32)/225

smoothed = cv2.filter2D(res,-1,kernel)


import scipy.misc
scipy.misc.imsave('removed_dead_im.jpg', smoothed)
scipy.misc.imsave('removed_dead_imm3.jpg', mask)



from PIL import Image
image = Image.open("removed_dead_im.jpg")
bg = image.getpixel((0,0))
width, height = image.size
bg_count = next(n for n,c in image.getcolors(width*height) if c==bg) 
img_count = width*height - bg_count
removed_dead_img_percent = (img_count*100.0) / (width*height)
removed_dead_img_percent

from PIL import Image
image = Image.open("filter_orig_im.jpg")
bg = image.getpixel((0,0))
width, height = image.size
bg_count = next(n for n,c in image.getcolors(width*height) if c==bg )
img_count = width*height - bg_count
filter_orig_img_percent = img_count*100.0/width/height
filter_orig_img_percent

Percentage_dead = (filter_orig_img_percent - removed_dead_img_percent) / (filter_orig_img_percent) * 100
Percentage_dead

