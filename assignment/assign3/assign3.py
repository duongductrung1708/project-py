import cv2
import numpy as np
import json
import os


class ArtworkQualityChecker:

    def __init__(self, image_path):

        self.image_path = image_path

        self.image = cv2.imread(
            image_path,
            cv2.IMREAD_UNCHANGED
        )

        if self.image is None:
            raise ValueError("Không tìm thấy ảnh")

        self.height, self.width = self.image.shape[:2]

        self.issues = []
        self.metrics = {}

        os.makedirs("./output", exist_ok=True)

    # =====================================================
    # BUILD MASK
    # =====================================================

    def build_mask(self):

        bgr = self.image[:, :, :3]

        bg_color = bgr[0, 0].astype(np.int16)

        dist = np.linalg.norm(
            bgr.astype(np.int16) - bg_color,
            axis=2
        )

        mask = np.where(
            dist > 20,
            255,
            0
        ).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        self.mask = mask

    # =====================================================
    # DETECT ARTWORK
    # =====================================================

    def detect_artwork(self):

        contours, _ = cv2.findContours(
            self.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            raise ValueError(
                "Artwork not found"
            )

        self.contours = contours

        all_points = np.vstack(contours)

        self.x, self.y, self.w, self.h = cv2.boundingRect(
            all_points
        )

        self.metrics["contour_count"] = len(
            contours
        )

        self.metrics["svg_points"] = int(
            sum(len(cnt) for cnt in contours)
        )

    # =====================================================
    # RESOLUTION
    # =====================================================

    def check_resolution(self):

        if self.width < 1000 or self.height < 1000:
            self.issues.append(
                "low_resolution"
            )

    # =====================================================
    # BLUR
    # =====================================================

    def check_blur(self):

        gray = cv2.cvtColor(
            self.image[:, :, :3],
            cv2.COLOR_BGR2GRAY
        )

        lap_var = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        self.metrics[
            "laplacian_variance"
        ] = round(
            lap_var,
            2
        )

        if lap_var < 100:
            self.issues.append(
                "blur_image"
            )

    # =====================================================
    # ARTWORK SIZE
    # =====================================================

    def check_artwork_size(self):

        artwork_ratio = (
            self.w * self.h
        ) / (
            self.width * self.height
        )

        self.metrics[
            "artwork_area_ratio"
        ] = round(
            artwork_ratio,
            4
        )

        if artwork_ratio < 0.05:
            self.issues.append(
                "artwork_too_small"
            )

    # =====================================================
    # TOUCH EDGE
    # =====================================================

    def check_touch_edge(self):

        margin = 10

        if (
            self.x <= margin
            or self.y <= margin
            or self.x + self.w >= self.width - margin
            or self.y + self.h >= self.height - margin
        ):
            self.issues.append(
                "artwork_touch_edge"
            )

    # =====================================================
    # CENTERING
    # =====================================================

    def check_centering(self):

        artwork_cx = self.x + self.w / 2
        artwork_cy = self.y + self.h / 2

        canvas_cx = self.width / 2
        canvas_cy = self.height / 2

        dx = abs(
            artwork_cx - canvas_cx
        )

        dy = abs(
            artwork_cy - canvas_cy
        )

        self.metrics[
            "center_offset_x"
        ] = round(dx, 2)

        self.metrics[
            "center_offset_y"
        ] = round(dy, 2)

        if dx > self.width * 0.20:
            self.issues.append(
                "off_center_horizontal"
            )

        if dy > self.height * 0.20:
            self.issues.append(
                "off_center_vertical"
            )

    # =====================================================
    # MARGIN
    # =====================================================

    def check_margin(self):

        left = self.x
        right = self.width - (
            self.x + self.w
        )

        top = self.y
        bottom = self.height - (
            self.y + self.h
        )

        self.metrics["margin_left"] = left
        self.metrics["margin_right"] = right
        self.metrics["margin_top"] = top
        self.metrics["margin_bottom"] = bottom

        horizontal_balance = (
            min(left, right)
            / max(left, right)
        )

        vertical_balance = (
            min(top, bottom)
            / max(top, bottom)
        )

        self.metrics[
            "horizontal_balance"
        ] = round(
            horizontal_balance,
            3
        )

        self.metrics[
            "vertical_balance"
        ] = round(
            vertical_balance,
            3
        )

        if horizontal_balance < 0.4:
            self.issues.append(
                "unbalanced_horizontal_margin"
            )

        if vertical_balance < 0.4:
            self.issues.append(
                "unbalanced_vertical_margin"
            )

    # =====================================================
    # FOREGROUND
    # =====================================================

    def check_foreground_ratio(self):

        ratio = (
            np.count_nonzero(
                self.mask
            )
            / self.mask.size
        )

        whitespace_ratio = 1 - ratio

        self.metrics[
            "foreground_ratio"
        ] = round(
            ratio,
            4
        )

        self.metrics[
            "whitespace_ratio"
        ] = round(
            whitespace_ratio,
            4
        )

        if ratio > 0.95:
            self.issues.append(
                "background_detection_failed"
            )

    # =====================================================
    # THIN LINE CHECK
    # =====================================================

    def check_thin_lines(self):

        dist = cv2.distanceTransform(
            self.mask,
            cv2.DIST_L2,
            5
        )

        thin_pixels = np.count_nonzero(
            (dist > 0) & (dist < 2)
        )

        foreground_pixels = np.count_nonzero(
            self.mask
        )

        thin_ratio = (
            thin_pixels /
            max(foreground_pixels, 1)
        )

        self.metrics[
            "thin_line_ratio"
        ] = round(
            thin_ratio,
            4
        )

        if thin_ratio > 0.30:
            self.issues.append(
                "thin_line_detected"
            )

    # =====================================================
    # SVG COMPLEXITY
    # =====================================================

    def check_complexity(self):

        points = self.metrics[
            "svg_points"
        ]

        if points > 50000:
            self.issues.append(
                "svg_too_complex"
            )

    # =====================================================
    # PNG TRANSPARENCY
    # =====================================================

    def check_alpha_channel(self):

        if (
            len(self.image.shape) == 3
            and self.image.shape[2] == 4
        ):

            alpha = self.image[:, :, 3]

            transparent_ratio = (
                np.count_nonzero(alpha == 0)
                / alpha.size
            )

            self.metrics[
                "transparent_ratio"
            ] = round(
                transparent_ratio,
                4
            )

    # =====================================================
    # DEBUG
    # =====================================================

    def save_debug(self):

        debug = self.image[:, :, :3].copy()

        cv2.rectangle(
            debug,
            (self.x, self.y),
            (self.x + self.w, self.y + self.h),
            (0, 255, 0),
            3
        )

        cv2.imwrite(
            "./output/debug_mask.png",
            self.mask
        )

        cv2.imwrite(
            "./output/debug_bbox.png",
            debug
        )

    # =====================================================
    # RUN
    # =====================================================

    def run(self):

        self.build_mask()

        self.detect_artwork()

        self.check_resolution()

        self.check_blur()

        self.check_artwork_size()

        self.check_touch_edge()

        self.check_centering()

        self.check_margin()

        self.check_foreground_ratio()

        self.check_thin_lines()

        self.check_complexity()

        self.check_alpha_channel()

        self.save_debug()

        result = {
            "status":
                "pass"
                if len(self.issues) == 0
                else "warning",

            "issues": self.issues,

            "metrics": {
                "width": self.width,
                "height": self.height,
                "artwork_width": self.w,
                "artwork_height": self.h,
                **self.metrics
            }
        }

        return result


# =====================================================
# MAIN
# =====================================================

checker = ArtworkQualityChecker(
    "../data/Croissant.jpg"
)

result = checker.run()

print("=" * 60)
print(
    json.dumps(
        result,
        indent=4
    )
)
print("=" * 60)