import cv2
import numpy as np
import cairosvg
from PIL import Image

image = cv2.imread('./data/bai1/exe_1.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, image_thresh = cv2.threshold(image_gray, 150, 255, cv2.THRESH_BINARY)
cv2.imshow('Binary image', image_thresh)

contours, hierarchy = cv2.findContours(image=image_thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(94, 83, 227), thickness=1, lineType=cv2.LINE_AA)

# Chuyển sang svg
commands = []

# Duyệt qua TẤT CẢ các contour tìm được
for contour in contours:
    first_x, first_y = contour[0][0]
    commands.append(f"M {first_x} {first_y}")
    for point in contour[1:]:
        x, y = point[0]
        commands.append(f"L {x} {y}")        
    commands.append("Z")

# Nối tất cả thành 1 chuỗi path
path_string = " ".join(commands)

# Ghi ra file SVG
with open("./data/bai1/exe_1_contours.svg", "w") as f:
    f.write(f'<svg viewBox="0 0 {image.shape[1]} {image.shape[0]}" xmlns="http://www.w3.org/2000/svg">\n <path d="{path_string}" fill="white" stroke="none"/>\n</svg>')

# Chuyển SVG thành PNG
cairosvg.svg2png(url="./data/bai1/exe_1_contours.svg", write_to="./data/bai1/exe_1_contours.png")

# Chuyển PNG thành BMP
img = Image.open("./data/bai1/exe_1_contours.png")
img.save("./data/bai1/exe_1_contours.bmp")

image_bmp = cv2.imread("./data/bai1/exe_1_contours.bmp", cv2.IMREAD_GRAYSCALE)
_, image_bmp_thresh = cv2.threshold(image_bmp, 127, 255, cv2.THRESH_BINARY)

# Tính IoU bằng Bitwise
mask_goc_bool = image_thresh.astype(bool)
mask_svg_bool = image_bmp_thresh.astype(bool)

intersection = np.logical_and(mask_goc_bool, mask_svg_bool).sum()
union = np.logical_or(mask_goc_bool, mask_svg_bool).sum()

iou = intersection / union
print(f"IoU: {iou * 100:.2f}%")

cv2.imshow('exe_1_contours.png', image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
