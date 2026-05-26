# Importing the libraries 
import cv2
import numpy as np

# Reading image
image = cv2.imread("./resource/cat.jpg")

# Get original height and width
h,w,c = image.shape
print("Original Height and Width:", h,"x", w)

# Set rows and columns
# lets downsize the image using new  width and height
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

# Set rows and columns
up_width = 600
up_height = 400
up_points = (up_width, up_height)
# resize the image
resized_up = cv2.resize(image, up_points, interpolation = cv2.INTER_LINEAR)

# Display images
# cv2.imshow('Resized Down by defining height and width', resized_down)
# cv2.waitKey()
# cv2.imshow('Resized Up image by defining height and width', resized_up)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Scaling Up the image 1.2 times by specifying both scaling factors
scale_up_x = 1.2
scale_up_y = 1.2
# Scaling Down the image 0.6 times specifying a single scale factor.
scale_down = 0.2
 
scaled_f_down = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
scaled_f_up = cv2.resize(image, None, fx= scale_up_x, fy= scale_up_y, interpolation= cv2.INTER_LINEAR)

# Display images and press any key to check next image
# cv2.imshow('Resized Down by defining scaling factor', scaled_f_down)
# cv2.waitKey()
# cv2.imshow('Resized Up image by defining scaling factor', scaled_f_up)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Scaling Down the image 0.6 times using different Interpolation Method
res_inter_nearest = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_NEAREST)
res_inter_linear = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_LINEAR)
res_inter_area = cv2.resize(image, None, fx= scale_down, fy= scale_down, interpolation= cv2.INTER_AREA)

# Concatenate images in horizontal axis for comparison
horizontal= np.concatenate((res_inter_nearest, res_inter_linear, res_inter_area), axis = 1)
# Display the image Press any key to continue
cv2.imshow('Inter Nearest :: Inter Linear :: Inter Area', horizontal)
cv2.waitKey()
cv2.destroyAllWindows()