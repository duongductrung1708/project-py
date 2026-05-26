	
# import the cv2 library
import cv2

img_color = cv2.imread("./resource/cat.jpg", cv2.IMREAD_COLOR)
img_gray = cv2.imread("./resource/cat.jpg", cv2.IMREAD_GRAYSCALE)
img_unchanged = cv2.imread("./resource/cat.jpg", cv2.IMREAD_UNCHANGED)

#Displays image inside a window
cv2.imshow('color image',img_color) 
cv2.imshow('grayscale image',img_gray)
cv2.imshow('unchanged image',img_unchanged)
 
# Waits for a keystroke
cv2.waitKey(0) 
 
# Destroys all the windows created
cv2.destroyAllWindows()

cv2.imwrite('grayscale.jpg',img_gray)
