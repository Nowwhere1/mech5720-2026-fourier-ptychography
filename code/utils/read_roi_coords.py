import rawpy
import cv2
import os

# ============================================================
# 只改这里：换成你要查看的 DNG 文件
# ============================================================
IMAGE_PATH = r"D:\FP_Data\iris_4mm\target_250um\iris4mm_250um_r04_c04.dng"


# ============================================================
# 读取 DNG
# ============================================================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Cannot find file:\n{IMAGE_PATH}")

with rawpy.imread(IMAGE_PATH) as raw:
    rgb = raw.postprocess(
        output_bps=8,
        gamma=(1, 1),
        no_auto_bright=False,
        use_camera_wb=True
    )

# OpenCV 使用 BGR
img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

h, w = img.shape[:2]

print(f"\nOriginal image size: {w} x {h}")
print("Left click anywhere to print the ORIGINAL image coordinate.")
print("Press Q or ESC to exit.\n")


# ============================================================
# 缩小显示，但保留原图坐标
# ============================================================
MAX_WIDTH = 1400
MAX_HEIGHT = 850

scale = min(
    MAX_WIDTH / w,
    MAX_HEIGHT / h,
    1.0
)

display_w = int(w * scale)
display_h = int(h * scale)

display_img = cv2.resize(
    img,
    (display_w, display_h),
    interpolation=cv2.INTER_AREA
)


# ============================================================
# 鼠标点击事件
# ============================================================
def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        # 把显示窗口坐标换算回原始DNG坐标
        original_x = round(x / scale)
        original_y = round(y / scale)

        print(f"x = {original_x}, y = {original_y}")

        # 在画面上标记点击位置
        cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)

        text = f"({original_x}, {original_y})"

        cv2.putText(
            display_img,
            text,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("ROI coordinate reader", display_img)


# ============================================================
# 显示窗口
# ============================================================
cv2.namedWindow("ROI coordinate reader", cv2.WINDOW_NORMAL)
cv2.imshow("ROI coordinate reader", display_img)

cv2.setMouseCallback(
    "ROI coordinate reader",
    mouse_callback
)

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()