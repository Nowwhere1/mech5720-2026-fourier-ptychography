import os
import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\FP_Data\iris_4mm"

# 建议先选一张条纹比较明显的，例如 r04_c05
IMAGE_PATH = os.path.join(
    BASE_DIR,
    "target_250um",
    "iris4mm_250um_r04_c05.dng"
)

AMBIENT_DIR = os.path.join(
    BASE_DIR,
    "ambient"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed_250um"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. EXTRACT R / G1 / G2 / B FROM ONE DNG
# ============================================================

def load_bayer_channels(filename):

    with rawpy.imread(filename) as raw:

        raw_img = raw.raw_image_visible.astype(np.float32)

        pattern = raw.raw_pattern
        desc = raw.color_desc.decode("ascii")

        positions = {
            "R": None,
            "B": None,
            "G": []
        }

        # Locate the 4 Bayer positions in the 2x2 CFA
        for r in range(2):
            for c in range(2):

                colour_index = pattern[r, c]
                colour = desc[colour_index]

                if colour == "R":
                    positions["R"] = (r, c)

                elif colour == "B":
                    positions["B"] = (r, c)

                elif colour == "G":
                    positions["G"].append((r, c))

        if positions["R"] is None:
            raise RuntimeError("Could not find Red Bayer pixel.")

        if positions["B"] is None:
            raise RuntimeError("Could not find Blue Bayer pixel.")

        if len(positions["G"]) != 2:
            raise RuntimeError(
                f"Expected 2 Green positions, found {len(positions['G'])}"
            )

        # Row-major order
        positions["G"] = sorted(positions["G"])

        rR, cR = positions["R"]
        rG1, cG1 = positions["G"][0]
        rG2, cG2 = positions["G"][1]
        rB, cB = positions["B"]

        channels = {
            "R": raw_img[rR::2, cR::2],
            "G1": raw_img[rG1::2, cG1::2],
            "G2": raw_img[rG2::2, cG2::2],
            "B": raw_img[rB::2, cB::2],
        }

        print("\nBayer pattern:")
        print(pattern)

        print("\nBayer positions:")
        print(f"R  : ({rR}, {cR})")
        print(f"G1 : ({rG1}, {cG1})")
        print(f"G2 : ({rG2}, {cG2})")
        print(f"B  : ({rB}, {cB})")

    return channels


# ============================================================
# 3. LOAD AND AVERAGE AMBIENT FRAMES
# ============================================================

ambient_files = sorted(
    glob.glob(
        os.path.join(
            AMBIENT_DIR,
            "*.dng"
        )
    )
)

if len(ambient_files) == 0:
    raise FileNotFoundError(
        f"No ambient DNG files found in:\n{AMBIENT_DIR}"
    )

print(f"\nFound {len(ambient_files)} ambient frames.")

ambient_stacks = {
    "R": [],
    "G1": [],
    "G2": [],
    "B": []
}

for filename in ambient_files:

    channels = load_bayer_channels(filename)

    for key in ambient_stacks:
        ambient_stacks[key].append(
            channels[key]
        )


ambient_avg = {}

for key in ambient_stacks:

    ambient_avg[key] = np.mean(
        np.stack(
            ambient_stacks[key],
            axis=0
        ),
        axis=0
    )


# ============================================================
# 4. LOAD TARGET IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Cannot find target image:\n{IMAGE_PATH}"
    )

target = load_bayer_channels(
    IMAGE_PATH
)


# ============================================================
# 5. AMBIENT CORRECTION
# ============================================================

corrected = {}

for key in target:

    corrected[key] = (
        target[key]
        -
        ambient_avg[key]
    )


# ============================================================
# 6. PRINT STATISTICS
# ============================================================

print()
print("=" * 78)
print("CHANNEL COMPARISON")
print("=" * 78)

print(
    f"{'Channel':<8}"
    f"{'Mean':>12}"
    f"{'Std':>12}"
    f"{'p99':>12}"
    f"{'p99.9':>12}"
    f"{'Max':>12}"
)

for key in ["R", "G1", "G2", "B"]:

    img = corrected[key]

    print(
        f"{key:<8}"
        f"{np.mean(img):>12.2f}"
        f"{np.std(img):>12.2f}"
        f"{np.percentile(img, 99):>12.2f}"
        f"{np.percentile(img, 99.9):>12.2f}"
        f"{np.max(img):>12.2f}"
    )

print("=" * 78)


# ============================================================
# 7. FIXED DISPLAY SCALE
#
# Same scale for all 4 channels:
# good for comparing actual signal strength.
# ============================================================

all_positive = np.concatenate(
    [
        corrected[key][corrected[key] > 0].ravel()
        for key in corrected
        if np.any(corrected[key] > 0)
    ]
)

FIXED_MIN = 0.0

FIXED_MAX = np.percentile(
    all_positive,
    99.8
)

print(
    f"\nShared fixed display range: "
    f"{FIXED_MIN:.2f} to {FIXED_MAX:.2f} DN"
)


# ============================================================
# 8. DRAW COMPARISON
# ============================================================

channels_order = [
    "R",
    "G1",
    "G2",
    "B"
]

fig, axes = plt.subplots(
    2,
    4,
    figsize=(16, 8)
)


# ------------------------------------------------------------
# TOP ROW:
# Same display scale
# ------------------------------------------------------------

for i, key in enumerate(channels_order):

    img = corrected[key]

    axes[0, i].imshow(
        img,
        cmap="gray",
        vmin=FIXED_MIN,
        vmax=FIXED_MAX
    )

    axes[0, i].set_title(
        f"{key} - fixed scale\n"
        f"p99.9 = {np.percentile(img, 99.9):.1f} DN"
    )

    axes[0, i].axis("off")


# ------------------------------------------------------------
# BOTTOM ROW:
# Auto scale each channel separately
# ------------------------------------------------------------

for i, key in enumerate(channels_order):

    img = corrected[key]

    local_min = np.percentile(
        img,
        1
    )

    local_max = np.percentile(
        img,
        99.8
    )

    axes[1, i].imshow(
        img,
        cmap="gray",
        vmin=local_min,
        vmax=local_max
    )

    axes[1, i].set_title(
        f"{key} - individual scale\n"
        f"{local_min:.1f} to {local_max:.1f} DN"
    )

    axes[1, i].axis("off")


fig.suptitle(
    "Bayer Channel Comparison\n"
    + os.path.basename(IMAGE_PATH)
    + "\nAmbient-corrected",
    fontsize=14
)

plt.tight_layout()

output_file = os.path.join(
    OUTPUT_DIR,
    "bayer_channel_comparison.png"
)

plt.savefig(
    output_file,
    dpi=200
)

plt.show()


# ============================================================
# 9. SAVE CHANNEL DATA
# ============================================================

for key in channels_order:

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"test_corrected_{key}.npy"
        ),
        corrected[key]
    )


print()
print("Done.")
print("Figure saved to:")
print(output_file)