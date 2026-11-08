import os
import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

TARGET_DIR = r"D:\FP_Data\iris_4mm\target_250um"

OUTPUT_FILE = r"D:\FP_Data\iris_4mm\raw_15_LED_montage.png"


# 你现在选定的 15 个 LED
LED_POSITIONS = [
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
]


# ============================================================
# FIND FILE
# ============================================================

def find_file(row, col):

    patterns = [
        os.path.join(
            TARGET_DIR,
            f"*r{row:02d}_c{col:02d}*.dng"
        ),
        os.path.join(
            TARGET_DIR,
            f"*r{row}_c{col}*.dng"
        ),
    ]

    matches = []

    for pattern in patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No DNG found for r{row:02d}_c{col:02d}"
        )

    return matches[0]


# ============================================================
# LOAD DNG AS RGB
# ============================================================

def load_rgb(filename):

    with rawpy.imread(filename) as raw:

        rgb = raw.postprocess(
            output_bps=16,
            gamma=(1, 1),          # linear
            no_auto_bright=True,   # 不让每张图自己自动变亮
            use_camera_wb=True
        )

    rgb = rgb.astype(np.float32)

    return rgb


# ============================================================
# LOAD ALL 15 IMAGES
# ============================================================

images = []
filenames = []

for row, col in LED_POSITIONS:

    filename = find_file(row, col)

    print(
        f"Loading r{row:02d}_c{col:02d}: "
        f"{os.path.basename(filename)}"
    )

    img = load_rgb(filename)

    images.append(img)
    filenames.append(filename)


images = np.stack(images, axis=0)


# ============================================================
# GLOBAL DISPLAY SCALE
#
# All images use the SAME scale so brightness differences
# between LEDs remain meaningful.
# ============================================================

DISPLAY_MAX = np.percentile(
    images,
    99.9
)

print()
print(
    f"Global display maximum = "
    f"{DISPLAY_MAX:.1f}"
)


display_images = np.clip(
    images / DISPLAY_MAX,
    0,
    1
)


# ============================================================
# CREATE 3 x 5 MONTAGE
# ============================================================

fig, axes = plt.subplots(
    3,
    5,
    figsize=(16, 9)
)

for i, ax in enumerate(axes.flat):

    row, col = LED_POSITIONS[i]

    ax.imshow(
        display_images[i]
    )

    ax.axis("off")

    # LED position BELOW image
    ax.text(
        0.5,
        -0.07,
        f"LED r{row:02d} c{col:02d}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11
    )


fig.suptitle(
    "Captured Images under Different Single-LED Illumination",
    fontsize=18,
    y=0.97
)

plt.subplots_adjust(
    left=0.03,
    right=0.97,
    top=0.90,
    bottom=0.08,
    wspace=0.10,
    hspace=0.28
)


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("Saved montage to:")
print(OUTPUT_FILE)