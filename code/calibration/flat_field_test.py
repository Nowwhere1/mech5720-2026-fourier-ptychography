import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. USER SETTINGS
# ============================================================

DATA_DIR = Path(r"D:\FP_Data\experiment2")

DARK_FILE = DATA_DIR / "dark.dng"
FLAT_FILE = DATA_DIR / "flat_r4c5.dng"
TARGET_FILE = DATA_DIR / "target_r4c5.dng"

OUTPUT_DIR = DATA_DIR / "flat_field_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Bayer channel:
# "R", "G1", "G2", or "B"
CHANNEL = "R"

# Prevent division by extremely dark flat pixels.
# This is deliberately much less aggressive than the previous 20% mask.
MIN_FLAT_FRACTION = 0.03

# Display percentiles
DISPLAY_LOW = 1
DISPLAY_HIGH = 99.5


# ============================================================
# 2. LOAD ONE RAW BAYER SUB-CHANNEL
# ============================================================

def load_bayer_channel(path, channel="R"):
    """
    Load visible RAW data and extract one Bayer sub-lattice.

    Supports:
        R
        G1
        G2
        B

    The Bayer position is determined from rawpy metadata rather than
    assuming a fixed RGGB/BGGR arrangement.
    """

    with rawpy.imread(str(path)) as raw:

        img = raw.raw_image_visible.astype(np.float32).copy()

        pattern = raw.raw_pattern
        color_desc = raw.color_desc.decode("ascii")

        print(f"\nFile: {path.name}")
        print("RAW Bayer pattern:")
        print(pattern)
        print("Color description:", color_desc)

        positions = {
            "R": [],
            "G": [],
            "B": []
        }

        for r in range(2):
            for c in range(2):
                color_index = pattern[r, c]
                color_name = color_desc[color_index]

                if color_name in positions:
                    positions[color_name].append((r, c))

        if channel == "R":
            r, c = positions["R"][0]

        elif channel == "B":
            r, c = positions["B"][0]

        elif channel == "G1":
            r, c = positions["G"][0]

        elif channel == "G2":
            r, c = positions["G"][1]

        else:
            raise ValueError(
                'CHANNEL must be "R", "G1", "G2", or "B"'
            )

        sub = img[r::2, c::2]

        print(
            f"Using channel {channel} at Bayer position "
            f"({r}, {c}), shape = {sub.shape}"
        )

        return sub


# ============================================================
# 3. DISPLAY NORMALIZATION
# ============================================================

def display_scale(img, low=1, high=99.5):
    """
    Robust normalization for display only.
    Does not modify the numerical image used for correction.
    """

    valid = img[np.isfinite(img)]

    if valid.size == 0:
        return np.zeros_like(img)

    vmin = np.percentile(valid, low)
    vmax = np.percentile(valid, high)

    if vmax <= vmin:
        vmax = vmin + 1.0

    out = (img - vmin) / (vmax - vmin)

    return np.clip(out, 0, 1)


# ============================================================
# 4. LOAD IMAGES
# ============================================================

print("\nLoading RAW images...")

dark = load_bayer_channel(DARK_FILE, CHANNEL)
flat = load_bayer_channel(FLAT_FILE, CHANNEL)
target = load_bayer_channel(TARGET_FILE, CHANNEL)


if not (dark.shape == flat.shape == target.shape):
    raise RuntimeError(
        f"Image sizes do not match:\n"
        f"dark   = {dark.shape}\n"
        f"flat   = {flat.shape}\n"
        f"target = {target.shape}"
    )


# ============================================================
# 5. DARK CORRECTION
# ============================================================

flat_signal = flat - dark
target_signal = target - dark

# For flat-field division, negative values are not meaningful.
# Keep original arrays if you want noise analysis later.
flat_signal_for_division = np.clip(flat_signal, 0, None)
target_signal_for_division = np.clip(target_signal, 0, None)

print("\nDark-corrected statistics:")

print(
    f"Flat:   min={flat_signal.min():.2f}, "
    f"mean={flat_signal.mean():.2f}, "
    f"max={flat_signal.max():.2f}"
)

print(
    f"Target: min={target_signal.min():.2f}, "
    f"mean={target_signal.mean():.2f}, "
    f"max={target_signal.max():.2f}"
)


# ============================================================
# 6. SELECT ROI INTERACTIVELY
# ============================================================

print("\n-------------------------------------------")
print("ROI selection")
print("-------------------------------------------")
print("A window will open.")
print("Click TWO points:")
print("1. top-left corner of the target region")
print("2. bottom-right corner of the target region")
print("-------------------------------------------\n")

fig = plt.figure(figsize=(9, 6))

plt.imshow(
    display_scale(target_signal),
    cmap="gray"
)

plt.title(
    "Click TOP-LEFT and BOTTOM-RIGHT corners of the ROI"
)

plt.axis("off")

points = plt.ginput(2, timeout=0)

plt.close(fig)


if len(points) != 2:
    raise RuntimeError("ROI selection cancelled.")


(x1, y1), (x2, y2) = points

x1, x2 = sorted([int(round(x1)), int(round(x2))])
y1, y2 = sorted([int(round(y1)), int(round(y2))])


# keep inside image
x1 = max(0, x1)
y1 = max(0, y1)
x2 = min(target.shape[1], x2)
y2 = min(target.shape[0], y2)


print(
    f"Selected ROI:\n"
    f"x = {x1}:{x2}\n"
    f"y = {y1}:{y2}"
)


# ============================================================
# 7. EXTRACT SAME ROI FROM ALL THREE IMAGES
# ============================================================

target_roi = target_signal_for_division[y1:y2, x1:x2]
flat_roi = flat_signal_for_division[y1:y2, x1:x2]
dark_roi = dark[y1:y2, x1:x2]


if target_roi.size == 0:
    raise RuntimeError("Selected ROI is empty.")


# ============================================================
# 8. FLAT-FIELD CORRECTION
# ============================================================

# Use a robust estimate of illumination level inside the chosen ROI.
positive_flat = flat_roi[flat_roi > 0]

if positive_flat.size == 0:
    raise RuntimeError(
        "Flat ROI contains no positive signal after dark subtraction."
    )


# Reference level = median illumination in ROI
flat_reference = np.median(positive_flat)

print(
    f"\nFlat reference level inside ROI = "
    f"{flat_reference:.3f}"
)


# Normalized illumination profile
flat_normalized = flat_roi / flat_reference


# Avoid unstable division in extremely dark areas
flat_floor = MIN_FLAT_FRACTION

flat_safe = np.maximum(
    flat_normalized,
    flat_floor
)


# Main correction:
#
# target ~= illumination × object
#
# corrected ~= target / illumination
#
corrected_roi = target_roi / flat_safe


# ============================================================
# 9. DIAGNOSTIC VALIDITY MASK
# ============================================================

valid_mask = flat_normalized >= MIN_FLAT_FRACTION

valid_fraction = 100 * np.mean(valid_mask)

print(
    f"Pixels above flat threshold: "
    f"{valid_fraction:.1f}%"
)


# Important:
# Do NOT turn invalid pixels white.
# Just leave the numerical result as-is for now.
# The validity mask is displayed separately.


# ============================================================
# 10. DISPLAY COMPARISON
# ============================================================

target_display = display_scale(
    target_roi,
    DISPLAY_LOW,
    DISPLAY_HIGH
)

flat_display = display_scale(
    flat_roi,
    DISPLAY_LOW,
    DISPLAY_HIGH
)

corrected_display = display_scale(
    corrected_roi,
    DISPLAY_LOW,
    DISPLAY_HIGH
)


fig, axes = plt.subplots(
    1,
    4,
    figsize=(16, 5)
)


axes[0].imshow(
    target_display,
    cmap="gray"
)
axes[0].set_title(
    "Target\nbefore correction"
)
axes[0].axis("off")


axes[1].imshow(
    flat_display,
    cmap="gray"
)
axes[1].set_title(
    "Flat\nillumination profile"
)
axes[1].axis("off")


axes[2].imshow(
    corrected_display,
    cmap="gray"
)
axes[2].set_title(
    "After per-LED\nflat-field correction"
)
axes[2].axis("off")


axes[3].imshow(
    valid_mask,
    cmap="gray",
    vmin=0,
    vmax=1
)
axes[3].set_title(
    "Valid illumination\nregion"
)
axes[3].axis("off")


plt.tight_layout()


comparison_path = OUTPUT_DIR / "flat_field_comparison.png"

plt.savefig(
    comparison_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. SAVE NUMERICAL DATA
# ============================================================

np.save(
    OUTPUT_DIR / "target_roi.npy",
    target_roi
)

np.save(
    OUTPUT_DIR / "flat_roi.npy",
    flat_roi
)

np.save(
    OUTPUT_DIR / "corrected_roi.npy",
    corrected_roi
)

np.save(
    OUTPUT_DIR / "valid_mask.npy",
    valid_mask
)


# Save display version too
plt.imsave(
    OUTPUT_DIR / "corrected_display.png",
    corrected_display,
    cmap="gray"
)


# ============================================================
# 12. PRINT SUMMARY
# ============================================================

print("\n===========================================")
print("Flat-field correction completed")
print("===========================================")

print(f"Channel: {CHANNEL}")

print(
    f"ROI size: "
    f"{target_roi.shape[1]} × {target_roi.shape[0]}"
)

print(
    f"Valid illuminated pixels: "
    f"{valid_fraction:.1f}%"
)

print(
    f"\nResults saved to:\n{OUTPUT_DIR}"
)

print(
    "\nMain file to inspect:\n"
    f"{comparison_path}"
)

print(
    "\nInterpretation:"
)

print(
    "If the illumination spot becomes much flatter but the "
    "bar deformation remains, the problem is not simply "
    "the non-uniform LED intensity profile."
)