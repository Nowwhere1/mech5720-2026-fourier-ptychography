import rawpy
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

# 改成你要测量的 DNG
IMAGE_PATH = r"D:\FP_Data\iris_4mm\target_250um\iris4mm_250um_r04_c04.dng"

# 点击多少根连续条纹
N_POINTS = 6

# 你的实际拍摄图里看起来是竖条纹，因此测 x 坐标
# 如果以后测的是横条纹，改成 "horizontal"
BARS = "vertical"

# Target定义：
# black width = 250 um
# clear gap   = 250 um
# 所以一个完整周期 = 500 um
TARGET_PERIOD_UM = 500.0


# ============================================================
# LOAD DNG AND EXTRACT GREEN G1
# ============================================================

with rawpy.imread(IMAGE_PATH) as raw:

    raw_img = raw.raw_image_visible.astype(np.float32)

    pattern = raw.raw_pattern
    desc = raw.color_desc.decode("ascii")

    # 找第一组 Green Bayer pixel
    green_pos = None

    for r in range(2):
        for c in range(2):

            colour_index = pattern[r, c]

            if desc[colour_index] == "G":
                green_pos = (r, c)
                break

        if green_pos is not None:
            break

    gr, gc = green_pos

    # Green G1 sublattice
    green = raw_img[gr::2, gc::2]


print(f"Green G1 image shape: {green.shape}")


# ============================================================
# DISPLAY SCALING
# ============================================================

# 只用于显示，不改变用于测量的像素坐标
vmin = np.percentile(green, 1)
vmax = np.percentile(green, 99.8)

plt.figure(figsize=(12, 8))

plt.imshow(
    green,
    cmap="gray",
    vmin=vmin,
    vmax=vmax
)

plt.title(
    f"Click {N_POINTS} consecutive bar edges\n"
    "Always click the SAME side of each bar"
)

plt.xlabel("x [Green G1 pixels]")
plt.ylabel("y [Green G1 pixels]")

print()
print("============================================")
print(f"Click {N_POINTS} consecutive black bars.")
print("Always click the SAME edge of every bar.")
print("For example: left edge, left edge, left edge...")
print("============================================")
print()


# ============================================================
# MANUAL CLICKS
# ============================================================

points = plt.ginput(N_POINTS, timeout=0)

plt.close()

points = np.array(points)

x = points[:, 0]
y = points[:, 1]


# ============================================================
# SELECT MEASUREMENT COORDINATE
# ============================================================

if BARS == "vertical":

    # Vertical bars repeat along x
    coordinate = x
    axis_name = "x"

elif BARS == "horizontal":

    # Horizontal bars repeat along y
    coordinate = y
    axis_name = "y"

else:
    raise ValueError("BARS must be 'vertical' or 'horizontal'")


# ============================================================
# LINEAR FIT
#
# coordinate = intercept + period * bar_index
# ============================================================

bar_index = np.arange(N_POINTS)

fit = np.polyfit(
    bar_index,
    coordinate,
    1
)

period_px = abs(fit[0])
intercept = fit[1]

fitted_coordinate = np.polyval(
    fit,
    bar_index
)

residuals = coordinate - fitted_coordinate

rms_error = np.sqrt(
    np.mean(residuals**2)
)


# ============================================================
# PIXEL SCALE
# ============================================================

pixel_size_um = TARGET_PERIOD_UM / period_px


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("============================================")
print("CLICKED POINTS")
print("============================================")

for i, (px, py) in enumerate(points):

    print(
        f"Bar {i+1}: "
        f"x = {px:.2f}, "
        f"y = {py:.2f}"
    )


print()
print("============================================")
print("CALIBRATION RESULT")
print("============================================")

print(
    f"Measured along {axis_name}-direction"
)

print(
    f"Average bar period = "
    f"{period_px:.3f} Green G1 pixels"
)

print(
    f"Fit RMS error = "
    f"{rms_error:.3f} pixels"
)

print(
    f"Physical target period = "
    f"{TARGET_PERIOD_UM:.1f} um"
)

print(
    f"Object-space pixel size = "
    f"{pixel_size_um:.4f} um/pixel"
)

print("============================================")


# ============================================================
# SHOW FIT
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    bar_index,
    coordinate,
    "o",
    label="Clicked positions"
)

plt.plot(
    bar_index,
    fitted_coordinate,
    "-",
    label="Linear fit"
)

plt.xlabel("Bar index")
plt.ylabel(f"{axis_name} coordinate [pixels]")

plt.title(
    f"Period = {period_px:.2f} px\n"
    f"Object pixel size = {pixel_size_um:.4f} um/px"
)

plt.grid()
plt.legend()

plt.show()