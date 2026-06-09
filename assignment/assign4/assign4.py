import cv2
import numpy as np
import cairosvg
import json
import os
import time

# =====================================================
# CONFIG
# =====================================================

IMAGE_PATH = "./output/croissant.png"

SVG_PATH = "./output/artwork.svg"
PNG_PATH = "./output/reconstructed.png"

OVERLAY_PATH = "./output/overlay.png"
REPORT_PATH = "./output/report.json"

os.makedirs("./output", exist_ok=True)

start_time = time.time()

# =====================================================
# LOAD IMAGE
# =====================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError("Không đọc được ảnh")

height, width = image.shape[:2]

# =====================================================
# BUILD ARTWORK MASK
# =====================================================

bg_color = image[0, 0].astype(np.int16)

dist = np.linalg.norm(
    image.astype(np.int16) - bg_color,
    axis=2
)

binary = np.where(
    dist > 20,
    255,
    0
).astype(np.uint8)

kernel = np.ones((3, 3), np.uint8)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel
)

binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

# =====================================================
# CONTOUR EXTRACTION
# =====================================================

contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_NONE
)

if len(contours) == 0:
    raise ValueError("Không tìm thấy contour")

# =====================================================
# SVG PATH GENERATION
# =====================================================

commands = []

total_points = 0

for contour in contours:

    if len(contour) < 3:
        continue

    total_points += len(contour)

    first_x, first_y = contour[0][0]

    commands.append(
        f"M {first_x} {first_y}"
    )

    for point in contour[1:]:
        x, y = point[0]

        commands.append(
            f"L {x} {y}"
        )

    commands.append("Z")

path_string = " ".join(commands)

# =====================================================
# SAVE SVG
# =====================================================

svg_content = f"""
<svg
xmlns="http://www.w3.org/2000/svg"
viewBox="0 0 {width} {height}"
width="{width}"
height="{height}"
>
<path
d="{path_string}"
fill="white"
stroke="white"
stroke-width="1"
fill-rule="evenodd"
/>
</svg>
"""

with open(
    SVG_PATH,
    "w",
    encoding="utf-8"
) as f:
    f.write(svg_content)

# =====================================================
# SVG -> PNG
# =====================================================

cairosvg.svg2png(
    url=SVG_PATH,
    write_to=PNG_PATH
)

# =====================================================
# RECONSTRUCT
# =====================================================

reconstructed = cv2.imread(
    PNG_PATH,
    cv2.IMREAD_GRAYSCALE
)

_, reconstructed_binary = cv2.threshold(
    reconstructed,
    250,
    255,
    cv2.THRESH_BINARY
)

# =====================================================
# IoU
# =====================================================

mask_original = binary.astype(bool)

mask_svg = reconstructed_binary.astype(bool)

intersection = np.logical_and(
    mask_original,
    mask_svg
).sum()

union = np.logical_or(
    mask_original,
    mask_svg
).sum()

iou = intersection / union

# =====================================================
# OVERLAY
# =====================================================

overlay = np.zeros(
    (height, width, 3),
    dtype=np.uint8
)

# đỏ = mất pixel

only_original = np.logical_and(
    binary > 0,
    reconstructed_binary == 0
)

overlay[only_original] = (0, 0, 255)

# xanh = pixel dư

only_svg = np.logical_and(
    binary == 0,
    reconstructed_binary > 0
)

overlay[only_svg] = (0, 255, 0)

cv2.imwrite(
    OVERLAY_PATH,
    overlay
)

# =====================================================
# REPORT
# =====================================================

processing_time = (
    time.time() - start_time
) * 1000

svg_size_kb = (
    os.path.getsize(SVG_PATH)
    / 1024
)

compression_ratio = (
    svg_size_kb
    /
    (
        os.path.getsize(IMAGE_PATH)
        / 1024
    )
)

report = {
    "status":
        "PASS"
        if iou >= 0.99
        else "FAIL",

    "iou":
        round(float(iou), 6),

    "contours":
        len(contours),

    "svg_points":
        total_points,

    "pixels_missing":
        int(
            only_original.sum()
        ),

    "pixels_added":
        int(
            only_svg.sum()
        ),

    "svg_file_size_kb":
        round(
            svg_size_kb,
            2
        ),

    "compression_ratio":
        round(
            compression_ratio,
            4
        ),

    "processing_time_ms":
        round(
            processing_time,
            2
        )
}

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )

# =====================================================
# OUTPUT
# =====================================================

print("=" * 60)
print("ARTWORK VECTORIZATION REPORT")
print("=" * 60)

print(
    json.dumps(
        report,
        indent=4
    )
)

print("=" * 60)
print("SVG:", SVG_PATH)
print("PNG:", PNG_PATH)
print("OVERLAY:", OVERLAY_PATH)
print("=" * 60)