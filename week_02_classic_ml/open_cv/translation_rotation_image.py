import cv2
import numpy as np
 
# Reading the image
image = cv2.imread('./resource/cat.jpg')
 
# dividing height and width by 2 to get the center of the image
height, width = image.shape[:2]

# --- 1. TỊNH TIẾN (TRANSLATION) ---
tx, ty = 500, 1000 # Dịch sang phải 50, xuống dưới 100
M_translate = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

# Dùng warpAffine để áp dụng ma trận
img_translated = cv2.warpAffine(image, M_translate, (width, height))

# get the center coordinates of the image to create the 2D rotation matrix
center = (width/2, height/2)
 
# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1)
 
# rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
 
cv2.imshow('Original image', image)
cv2.imshow('Translated image', img_translated)
cv2.imshow('Rotated image', rotated_image)
# wait indefinitely, press any key on keyboard to exit
cv2.waitKey(0)
# save the rotated image to disk
cv2.imwrite('rotated_image.jpg', rotated_image)