# import cv2
# import numpy as np

# image = cv2.imread('../data/kodak.jpg') 
# if image is None:
#     raise ValueError("Không tìm thấy ảnh! Hãy kiểm tra lại đường dẫn.")

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Tách nền trắng: Giả sử nền luôn sáng (>240). 
# # Các chi tiết tối hơn nền (<240) sẽ được gán là 255 (trắng - dự đoán là logo)
# _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# # Dùng Morphology đóng các khoảng hở nhỏ để mask liền mạch hơn
# kernel = np.ones((5, 5), np.uint8)
# thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# mask = np.zeros(image.shape[:2], np.uint8)

# # 3.1: Phủ toàn bộ ảnh mặc định là "Có thể là nền" (PR_BGD = 2)
# mask[:] = cv2.GC_PR_BGD 

# # 3.2: Nơi nào Threshold bắt được (logo), ta gán là "Có thể là tiền cảnh" (PR_FGD = 3)
# mask[thresh == 255] = cv2.GC_PR_FGD

# # 3.3: Chốt chặn an toàn - Viền mép ngoài cùng của ảnh "Chắc chắn là nền" (BGD = 0).
# # Điều này giúp GrabCut lấy mẫu màu nền chuẩn xác nhất.
# mask[0:5, :] = cv2.GC_BGD
# mask[-5:, :] = cv2.GC_BGD
# mask[:, 0:5] = cv2.GC_BGD
# mask[:, -5:] = cv2.GC_BGD

# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# # Chạy GrabCut dựa trên mask mồi
# cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

# # Tạo Binary Mask: Lấy phần "Chắc chắn là tiền cảnh" (1) và "Có thể là tiền cảnh" (3)
# mask_binary = np.where((mask == 1) | (mask == 3), 1, 0).astype('uint8')

# # ==========================================aimport cv2
# import numpy as np

# # ==========================================
# # Bước 1: Đọc ảnh đầu vào
# # ==========================================
# image_path = '../data/kodak.jpg' # Đổi lại đường dẫn nếu cần
# image = cv2.imread(image_path)
# if image is None:
#     raise ValueError(f"Không tìm thấy ảnh tại '{image_path}'. Hãy kiểm tra lại đường dẫn!")

# # ==========================================
# # Bước 2: Khử nhiễu nền & Tạo Silhouette Mask (CẬP NHẬT CHO NỀN TRẮNG)
# # ==========================================
# # Làm mờ ảnh để khử nhiễu hạt
# blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
# gray = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

# # LẬT NGƯỢC LOGIC: Ảnh gốc nền trắng (> 240). 
# # Dùng THRESH_BINARY_INV để biến nền trắng thành đen (0), và logo tối màu thành trắng (255)
# _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# # Đóng các khoảng hở nhỏ để nét đứt được liền mạch
# kernel = np.ones((7, 7), np.uint8)
# thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# # Kỹ thuật Silhouette: Tìm viền ngoài cùng và TÔ ĐẶC không gian bên trong
# contours_thresh, _ = cv2.findContours(thresh_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# silhouette = np.zeros_like(gray)
# if contours_thresh:
#     cv2.drawContours(silhouette, contours_thresh, -1, 255, thickness=cv2.FILLED)

# # ==========================================
# # Bước 3: Thiết lập Mask thông minh cho GrabCut
# # ==========================================
# # Bóp nhỏ Silhouette lại 15 pixel để lấy cái "Lõi" chắc chắn là logo
# kernel_erode = np.ones((15, 15), np.uint8)
# sure_fg = cv2.erode(silhouette, kernel_erode, iterations=1)

# mask = np.zeros(image.shape[:2], np.uint8)

# # Phân bố các nhãn vùng cho thuật toán học:
# mask[:] = cv2.GC_PR_BGD                 # Toàn bộ ảnh: Có thể là nền (2)
# mask[silhouette == 255] = cv2.GC_PR_FGD # Vỏ bao quanh: Có thể là tiền cảnh (3)
# mask[sure_fg == 255] = cv2.GC_FGD       # Lõi bên trong: CHẮC CHẮN LÀ TIỀN CẢNH (1)
# # (Không đặt GC_BGD ở viền ảnh để cứu các góc sáng bị dính sát mép)

# # ==========================================
# # Bước 4: Chạy GrabCut
# # ==========================================
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)

# cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

# # Tạo Binary Mask: Lấy phần "Chắc chắn" (1) và "Có thể" (3)
# mask_binary = np.where((mask == 1) | (mask == 3), 1, 0).astype('uint8')

# # ==========================================
# # Bước 5: Khử răng cưa (Anti-aliasing)
# # ==========================================
# # Chuyển đổi sang số thực để chứa các điểm ảnh bán trong suốt
# mask_float = mask_binary.astype(np.float32)

# # Làm mờ viền mask bằng GaussianBlur (Feathering)
# mask_smoothed = cv2.GaussianBlur(mask_float, (5, 5), 0)

# # ==========================================
# # Bước 6: Ghép kênh Alpha & Xuất file
# # ==========================================
# image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

# # Gán kênh Alpha (Nhân 255 và ép kiểu về uint8)
# image_bgra[:, :, 3] = (mask_smoothed * 255).astype(np.uint8)

# cv2.imwrite('./output/kodak_transparent.png', image_bgra)
# print("✅ Pipeline hoàn tất! Đã lưu file: kodak_transparent.png")

# # ==========================================
# # Bước 7: Tính toán IoU Đánh giá KPI
# # ==========================================
# def calculate_iou(mask_pred, mask_gt):
#     # Đưa về Binary: Chỉ coi những pixel có độ đục > 50% (giá trị > 127) là phần thân
#     pred_binary = (mask_pred > 127).astype(np.uint8) 
#     gt_binary = (mask_gt > 127).astype(np.uint8)
    
#     intersection = np.logical_and(pred_binary, gt_binary).sum()
#     union = np.logical_or(pred_binary, gt_binary).sum()
    
#     if union == 0:
#         return 1.0 if intersection == 0 else 0.0
#     return intersection / union

# try:
#     # Đọc Ground Truth (đảm bảo file này đã được tạo ra từ script lấy Alpha channel của web)
#     mask_ground_truth = cv2.imread('../data/kodak_ground_truth.png', cv2.IMREAD_GRAYSCALE)
    
#     if mask_ground_truth is not None:
#         mask_ground_truth = cv2.resize(mask_ground_truth, (image.shape[1], image.shape[0]))
        
#         # Tạo mask 8-bit (0-255) từ mask_smoothed để đưa vào tính IoU
#         mask_pred_255 = (mask_smoothed * 255).astype(np.uint8)
#         iou_score = calculate_iou(mask_pred_255, mask_ground_truth)
        
#         print("-" * 30)
#         print(f"🎯 Chỉ số Foreground IoU: {iou_score * 100:.2f}%")
        
#         if iou_score >= 0.98:
#             print("🏆 XUẤT SẮC: Vượt KPI mức Chuyên gia (> 98%)")
#         elif iou_score >= 0.95:
#             print("✅ ĐẠT KPI (> 95%)")
#         else:
#             print("❌ KHÔNG ĐẠT KPI")
# except Exception as e:
#     print(f"⚠️ Không thể tính IoU do thiếu file Ground Truth: {e}")

# # Bước 4: Vá lỗi lỗ thủng và CHỐNG LẸM VIỀN (Dilation)
# # ==========================================
# contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# mask_final = np.zeros_like(mask_binary)

# if contours:
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 150: 
#             cv2.drawContours(mask_final, [cnt], -1, 1, thickness=cv2.FILLED)

# # CỨU PHẦN BỊ LẸM: Nong rộng mask ra 1 chút để lấy lại phần đáy màu trắng bị GrabCut gọt mất
# # kernel_dilate = np.ones((3, 3), np.uint8)
# # mask_final = cv2.dilate(mask_final, kernel_dilate, iterations=1)

# # ==========================================
# # Bước 5: Khử răng cưa (Anti-aliasing) & Output
# # ==========================================
# # 5.1 Chuyển mask sang dạng số thực (float) để có thể chứa các giá trị bán trong suốt
# mask_float = mask_final.astype(np.float32)

# # 5.2 Làm mờ viền mask (Gaussian Blur). 
# # Bộ lọc (5, 5) sẽ vuốt mềm các bậc thang răng cưa thành đường cong mịn.
# mask_smoothed = cv2.GaussianBlur(mask_float, (5, 5), 0)

# # 5.3 Gán kênh Alpha và xuất file
# image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
# # Ép kiểu về lại uint8 (0-255) cho ảnh màu
# image_bgra[:, :, 3] = (mask_smoothed * 255).astype(np.uint8)

# cv2.imwrite('./output/kodak_transparent.png', image_bgra)
# print("Đã xử lý xong: Chống lẹm viền và Khử răng cưa thành công!")

# def calculate_iou(mask_pred, mask_gt):
#     """
#     Hàm tính toán IoU giữa mask dự đoán (thuật toán) và mask chuẩn (Ground Truth).
#     """
#     # 1. Đảm bảo cả hai mảng đều là dạng Binary nguyên thủy (chỉ có 0 và 1)
#     pred_binary = (mask_pred > 0).astype(np.uint8)
#     gt_binary = (mask_gt > 127).astype(np.uint8)
    
#     # 2. Tính phần giao (Intersection) - Điểm pixel mà CẢ HAI đều là 1
#     intersection = np.logical_and(pred_binary, gt_binary).sum()
    
#     # 3. Tính phần hợp (Union) - Điểm pixel mà ÍT NHẤT MỘT TRONG HAI là 1
#     union = np.logical_or(pred_binary, gt_binary).sum()
    
#     # Tránh lỗi chia cho 0 trong trường hợp cả 2 ảnh đều toàn màu đen
#     if union == 0:
#         return 1.0 if intersection == 0 else 0.0
        
#     # 4. Công thức IoU
#     iou = intersection / union
#     return iou

# # Giả sử bạn có file nhãn chuẩn (do người dùng tự tô màu trắng cho logo, nền đen)
# # LƯU Ý: Phải đọc dưới dạng ảnh xám (IMREAD_GRAYSCALE)
# try:
#     mask_ground_truth = cv2.imread('./output/kodak_ground_truth.png', cv2.IMREAD_GRAYSCALE)
    
#     if mask_ground_truth is not None:
#         # Resize ground truth cho khớp với ảnh dự đoán (nếu cần thiết)
#         mask_ground_truth = cv2.resize(mask_ground_truth, (mask_final.shape[1], mask_final.shape[0]))
        
#         # Gọi hàm tính IoU. Biến mask_final được lấy từ Bước 4 ở script trước.
#         iou_score = calculate_iou(mask_final, mask_ground_truth)
        
#         print("-" * 30)
#         print(f"Chỉ số Foreground IoU: {iou_score * 100:.2f}%")
        
#         if iou_score >= 0.95:
#             print("ĐẠT KPI (> 95%)")
#         else:
#             print("KHÔNG ĐẠT KPI")
            
# except Exception as e:
#     print(f"Không thể tính IoU do thiếu file Ground Truth: {e}")

import cv2
import numpy as np
# import cairosvg

# =====================================================
# 1. READ IMAGE
# =====================================================

image = cv2.imread("../data/Croissant.jpg")

if image is None:
    raise Exception("Không đọc được ảnh")

h, w = image.shape[:2]

# =====================================================
# 2. BINARY
# =====================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)

kernel = np.ones((3, 3), np.uint8)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

cv2.imwrite("./output/binary.png", binary)

# =====================================================
# 3. FIND CONTOURS
# =====================================================

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

filtered = []

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 50:
        continue

    epsilon = 0.002 * cv2.arcLength(cnt, True)

    approx = cv2.approxPolyDP(
        cnt,
        epsilon,
        True
    )

    filtered.append(approx)

print("Contours:", len(filtered))

# =====================================================
# 4. DEBUG CONTOUR
# =====================================================

debug = image.copy()

cv2.drawContours(
    debug,
    filtered,
    -1,
    (0, 0, 255),
    1
)

cv2.imwrite("./output/contours.png", debug)

# =====================================================
# 5. SVG PATH
# =====================================================

commands = []

for contour in filtered:

    first_x, first_y = contour[0][0]

    commands.append(
        f"M {first_x + 0.25} {first_y + 0.25}"
    )

    for point in contour[1:]:

        x, y = point[0]

        commands.append(
            f"L {x + 0.25} {y + 0.25}"
        )

    commands.append("Z")

path_string = " ".join(commands)

svg_content = f"""
<svg xmlns="http://www.w3.org/2000/svg"
     width="{w}"
     height="{h}"
     viewBox="0 0 {w} {h}">

    <rect width="100%" height="100%" fill="black"/>

    <path
        d="{path_string}"
        fill="white"
        stroke="none"
        shape-rendering="crispEdges"
    />
</svg>
"""

with open("./output/logo.svg", "w") as f:
    f.write(svg_content)

print("SVG saved")

# =====================================================
# 6. SVG -> PNG
# =====================================================

# cairosvg.svg2png(
#     url="./output/logo.svg",
#     write_to="./output/render.png"
# )

# =====================================================
# 7. LOAD RENDER
# =====================================================

render = cv2.imread(
    "./output/render.png",
    cv2.IMREAD_GRAYSCALE
)

_, render_binary = cv2.threshold(
    render,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    "./output/render_binary.png",
    render_binary
)

# =====================================================
# 8. IOU
# =====================================================

mask_original = binary > 0
mask_render = render_binary > 0

intersection = np.logical_and(
    mask_original,
    mask_render
).sum()

union = np.logical_or(
    mask_original,
    mask_render
).sum()

iou = intersection / union

print()
print("=" * 50)
print(f"IoU = {iou * 100:.4f}%")
print("=" * 50)

# =====================================================
# 9. OVERLAY
# =====================================================

overlay = np.zeros(
    (h, w, 3),
    dtype=np.uint8
)

# Đỏ: mất sau vector hóa
only_original = np.logical_and(
    mask_original,
    ~mask_render
)

# Xanh lá: SVG thừa
only_svg = np.logical_and(
    ~mask_original,
    mask_render
)

overlay[only_original] = (0, 0, 255)
overlay[only_svg] = (0, 255, 0)

cv2.imwrite(
    "./output/overlay.png",
    overlay
)

print("Overlay saved")