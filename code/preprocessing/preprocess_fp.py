import os
import glob
import json
import rawpy
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"D:\FP_Data\iris_4mm"

TARGET_DIR = os.path.join(BASE_DIR, "target_250um")
AMBIENT_DIR = os.path.join(BASE_DIR, "ambient")
DARK_DIR = os.path.join(BASE_DIR, "dark_frame")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed_250um")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LED POSITIONS
#    First test: 3 rows x 5 columns = 15 images
# ============================================================

LED_POSITIONS = [
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
]

CENTRAL_LED = (4, 4)


# ============================================================
# 3. READ ONE DNG AND EXTRACT GREEN G1
# ============================================================

def load_green_g1(filename):
    """
    Read RAW DNG and extract the first green Bayer sublattice (G1).

    No demosaicing.
    No gamma.
    No auto brightness.

    Output:
        2D float32 image in original RAW DN.
    """

    with rawpy.imread(filename) as raw:

        raw_img = raw.raw_image_visible.astype(np.float32)

        pattern = raw.raw_pattern
        desc = raw.color_desc.decode("ascii")

        green_pos = None

        # Find first Green location in 2x2 Bayer pattern
        for r in range(2):
            for c in range(2):

                colour_index = pattern[r, c]

                if desc[colour_index] == "G":
                    green_pos = (r, c)
                    break

            if green_pos is not None:
                break

        if green_pos is None:
            raise RuntimeError(
                f"Could not find Green channel in {filename}"
            )

        gr, gc = green_pos

        green = raw_img[gr::2, gc::2]

    return green


# ============================================================
# 4. FIND TARGET FILE FROM LED POSITION
# ============================================================

def find_target_file(row, col):

    # Expected style:
    # iris4mm_250um_r03_c02.dng

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

    # Remove duplicates
    matches = sorted(set(matches))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No DNG found for r{row:02d}_c{col:02d}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple files found for "
            f"r{row:02d}_c{col:02d}"
        )

        for m in matches:
            print("   ", m)

        print("Using:", matches[0])

    return matches[0]


# ============================================================
# 5. LOAD AND AVERAGE CALIBRATION FRAMES
# ============================================================

def load_average_folder(folder):

    files = sorted(
        glob.glob(os.path.join(folder, "*.dng"))
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"No DNG files found in:\n{folder}"
        )

    print()
    print(f"Loading {len(files)} frames from:")
    print(folder)

    images = []

    for filename in files:

        print(
            "   ",
            os.path.basename(filename)
        )

        img = load_green_g1(filename)
        images.append(img)

    stack = np.stack(images, axis=0)

    average = np.mean(stack, axis=0)

    return average, files


# ============================================================
# 6. LOAD DARK + AMBIENT
# ============================================================

print()
print("==========================================")
print("Loading calibration images")
print("==========================================")

dark_avg, dark_files = load_average_folder(
    DARK_DIR
)

ambient_avg, ambient_files = load_average_folder(
    AMBIENT_DIR
)

if dark_avg.shape != ambient_avg.shape:
    raise RuntimeError(
        "Dark and ambient image sizes do not match."
    )

print()
print("Green G1 image shape:", dark_avg.shape)

print(
    f"Mean dark level: "
    f"{np.mean(dark_avg):.3f} DN"
)

print(
    f"Mean ambient level: "
    f"{np.mean(ambient_avg):.3f} DN"
)

ambient_light = ambient_avg - dark_avg

print(
    f"Mean ambient contribution above dark: "
    f"{np.mean(ambient_light):.3f} DN"
)


# ============================================================
# 7. LOAD 15 TARGET IMAGES
# ============================================================

print()
print("==========================================")
print("Loading target images")
print("==========================================")

raw_stack = []
corrected_stack = []
filenames = []

for row, col in LED_POSITIONS:

    filename = find_target_file(
        row,
        col
    )

    print(
        f"r{row:02d}_c{col:02d}: "
        f"{os.path.basename(filename)}"
    )

    target = load_green_g1(filename)

    if target.shape != dark_avg.shape:
        raise RuntimeError(
            f"Image shape mismatch:\n{filename}"
        )

    # --------------------------------------------------------
    # Full physical correction:
    #
    # target_dark_corrected = target - dark
    #
    # ambient_light = ambient - dark
    #
    # corrected =
    # (target - dark) - (ambient - dark)
    #
    # which simplifies to:
    #
    # corrected = target - ambient
    #
    # IMPORTANT:
    # Do NOT clamp negative values here.
    # --------------------------------------------------------

    corrected = target - ambient_avg

    raw_stack.append(target)
    corrected_stack.append(corrected)
    filenames.append(filename)


raw_stack = np.stack(
    raw_stack,
    axis=0
).astype(np.float32)

corrected_stack = np.stack(
    corrected_stack,
    axis=0
).astype(np.float32)


# ============================================================
# 8. BASIC STATISTICS
# ============================================================

print()
print("==========================================")
print("Corrected image statistics")
print("==========================================")

for i, (row, col) in enumerate(LED_POSITIONS):

    img = corrected_stack[i]

    print(
        f"r{row:02d}_c{col:02d}  "
        f"mean={np.mean(img):8.3f} DN   "
        f"std={np.std(img):8.3f} DN   "
        f"p99.9={np.percentile(img, 99.9):8.3f} DN"
    )


# ============================================================
# 9. DISPLAY DARK AND AMBIENT
# ============================================================

fig = plt.figure(figsize=(8, 6))

plt.imshow(
    dark_avg,
    cmap="gray"
)

plt.title(
    f"Average Dark Frame\n"
    f"{len(dark_files)} frames"
)

plt.colorbar(label="RAW DN")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_average_dark.png"
    ),
    dpi=200
)

plt.show()


fig = plt.figure(figsize=(8, 6))

plt.imshow(
    ambient_avg,
    cmap="gray"
)

plt.title(
    f"Average Ambient Frame\n"
    f"{len(ambient_files)} frames"
)

plt.colorbar(label="RAW DN")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_average_ambient.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 10. DISPLAY AMBIENT CONTRIBUTION
# ============================================================

fig = plt.figure(figsize=(8, 6))

plt.imshow(
    ambient_light,
    cmap="gray"
)

plt.title(
    "Ambient Contribution\n"
    "(Ambient - Dark)"
)

plt.colorbar(label="DN above dark")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_ambient_minus_dark.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 11. FIXED DISPLAY SCALE FOR CORRECTED IMAGES
#
# Display only.
# Negative values are preserved in corrected_stack.
# ============================================================

positive_values = corrected_stack[
    corrected_stack > 0
]

if len(positive_values) == 0:
    raise RuntimeError(
        "All corrected values are <= 0. "
        "Check ambient calibration."
    )

DISPLAY_MIN = 0

DISPLAY_MAX = np.percentile(
    positive_values,
    99.8
)

print()
print(
    f"Fixed montage display scale: "
    f"{DISPLAY_MIN:.2f} to {DISPLAY_MAX:.2f} DN"
)


# ============================================================
# 12. 3 x 5 CORRECTED MONTAGE
# ============================================================

fig, axes = plt.subplots(
    3,
    5,
    figsize=(15, 9)
)

for i, ax in enumerate(axes.flat):

    row, col = LED_POSITIONS[i]
    img = corrected_stack[i]

    ax.imshow(
        img,
        cmap="gray",
        vmin=DISPLAY_MIN,
        vmax=DISPLAY_MAX
    )

    p999 = np.percentile(
        img,
        99.9
    )

    ax.set_title(
        f"r{row:02d} c{col:02d}\n"
        f"p99.9={p999:.1f} DN",
        fontsize=9
    )

    ax.axis("off")


fig.suptitle(
    "4 mm aperture / 250 um target\n"
    "Green G1 / Ambient-corrected / Fixed display scale",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_corrected_15_LED_montage.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 13. CENTRAL LED BASELINE
# ============================================================

central_index = LED_POSITIONS.index(
    CENTRAL_LED
)

central_raw = raw_stack[
    central_index
]

central_corrected = corrected_stack[
    central_index
]


# Raw G1 baseline
fig = plt.figure(figsize=(9, 7))

plt.imshow(
    central_raw,
    cmap="gray"
)

plt.title(
    "Central LED r04_c04\n"
    "Raw Green G1"
)

plt.colorbar(label="RAW DN")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_central_raw_G1.png"
    ),
    dpi=200
)

plt.show()


# Corrected baseline
fig = plt.figure(figsize=(9, 7))

plt.imshow(
    central_corrected,
    cmap="gray",
    vmin=DISPLAY_MIN,
    vmax=DISPLAY_MAX
)

plt.title(
    "Conventional Baseline\n"
    "Central LED r04_c04 / Ambient-corrected"
)

plt.colorbar(
    label="Corrected DN"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "06_central_corrected_baseline.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 14. SAVE NUMERICAL DATA FOR FP
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "corrected_stack.npy"
    ),
    corrected_stack
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "raw_stack.npy"
    ),
    raw_stack
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "average_dark.npy"
    ),
    dark_avg
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "average_ambient.npy"
    ),
    ambient_avg
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "central_corrected_baseline.npy"
    ),
    central_corrected
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "led_positions.npy"
    ),
    np.array(LED_POSITIONS)
)


# ============================================================
# 15. SAVE METADATA
# ============================================================

metadata = {

    "aperture_mm": 4.0,

    "target": "250um",

    "channel": "Green G1",

    "LED_positions": [
        [int(r), int(c)]
        for r, c in LED_POSITIONS
    ],

    "central_LED": [
        CENTRAL_LED[0],
        CENTRAL_LED[1]
    ],

    "number_of_target_frames": int(
        len(LED_POSITIONS)
    ),

    "number_of_dark_frames": int(
        len(dark_files)
    ),

    "number_of_ambient_frames": int(
        len(ambient_files)
    ),

    "correction": (
        "corrected = target - average_ambient; "
        "average_dark retained separately for diagnostics"
    ),

    "display_min_DN": float(
        DISPLAY_MIN
    ),

    "display_max_DN": float(
        DISPLAY_MAX
    )
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    ),
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ============================================================
# DONE
# ============================================================

print()
print("==========================================")
print("PREPROCESSING COMPLETE")
print("==========================================")

print()
print("Output folder:")
print(OUTPUT_DIR)

print()
print("Saved:")
print("  corrected_stack.npy")
print("  raw_stack.npy")
print("  average_dark.npy")
print("  average_ambient.npy")
print("  central_corrected_baseline.npy")
print("  led_positions.npy")
print("  metadata.json")
print("  diagnostic PNG figures")

print()
print(
    "Next step: use corrected_stack.npy "
    "for Fourier Ptychography reconstruction."
)