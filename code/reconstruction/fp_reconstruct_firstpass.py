import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json


# ============================================================
# 1. PATH
# ============================================================

DATA_DIR = r"D:\FP_Data\experiment1\iris_4mm\processed_250um"

STACK_PATH = os.path.join(
    DATA_DIR,
    "corrected_stack.npy"
)

LED_PATH = os.path.join(
    DATA_DIR,
    "led_positions.npy"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "fp_firstpass"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. CURRENT PARAMETERS
# ============================================================

WAVELENGTH_NM = 630.0

LED_PITCH_MM = 4.0
LED_DISTANCE_MM = 80.0

CENTER_ROW = 4
CENTER_COL = 4

# First-pass model parameter.
# Do NOT treat this as a measured NA yet.
EFFECTIVE_NA = 0.067

# Number of complete passes through all LED measurements
N_ITERATIONS = 10


# ============================================================
# 3. LOAD DATA
# ============================================================

stack = np.load(STACK_PATH).astype(np.float32)

led_positions = np.load(LED_PATH)

print("Input stack:", stack.shape)
print("LED positions:", led_positions.shape)

n_images, H, W = stack.shape


# ============================================================
# 4. CHOOSE ONE FIXED ROI
# ============================================================

# Use r04_c05 because the bars were clearly visible there.
reference_led = (4, 5)

reference_index = None

for i, (r, c) in enumerate(led_positions):

    if int(r) == reference_led[0] and int(c) == reference_led[1]:

        reference_index = i
        break


if reference_index is None:
    raise RuntimeError("Reference LED not found.")


reference = stack[reference_index]


# Display-only scaling
vmin = np.percentile(reference, 1)
vmax = np.percentile(reference, 99.8)

display = np.clip(
    (reference - vmin) / (vmax - vmin),
    0,
    1
)

display8 = (display * 255).astype(np.uint8)


print()
print("Draw ONE box around the target.")
print("Leave some dark margin around the bars.")
print("Try roughly 100-200 pixels wide/high.")
print("Press ENTER when finished.")

x, y, w, h = cv2.selectROI(
    "Select common FP ROI",
    display8,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

x = int(x)
y = int(y)
w = int(w)
h = int(h)

print(
    f"ROI: x={x}, y={y}, "
    f"width={w}, height={h}"
)

# ============================================================
# SAVE ROI FOR REPRODUCIBILITY
# ============================================================

roi_info = {
    "x": x,
    "y": y,
    "width": w,
    "height": h
}

roi_path = os.path.join(
    DATA_DIR,
    "roi.json"
)

with open(roi_path, "w") as f:
    json.dump(
        roi_info,
        f,
        indent=4
    )

print("ROI saved to:")
print(roi_path)
# ============================================================
# 5. APPLY THE SAME ROI TO ALL 15 IMAGES
# ============================================================

roi_stack = stack[
    :,
    y:y+h,
    x:x+w
].copy()

print(
    "ROI stack shape:",
    roi_stack.shape
)


# ============================================================
# 6. PREPARE MEASURED INTENSITIES
# ============================================================

# Background subtraction can produce small negative values.
# Keep them in preprocessing, but intensity passed into sqrt()
# must physically be >= 0.
intensity_stack = np.maximum(
    roi_stack,
    0.0
)


# Estimate one scalar illumination strength for each LED.
# This compensates approximately for LED brightness / vignetting.
frame_scales = []

for img in intensity_stack:

    scale = np.percentile(
        img,
        99.5
    )

    frame_scales.append(scale)

frame_scales = np.array(frame_scales)


print()
print("Frame intensity scales:")

for i, (r, c) in enumerate(led_positions):

    print(
        f"r{int(r):02d}_c{int(c):02d}: "
        f"{frame_scales[i]:.3f}"
    )


# Avoid dividing noise-only frames by tiny numbers
valid = frame_scales > 1.0

print()
print(
    f"Frames retained: "
    f"{np.sum(valid)} / {len(valid)}"
)


intensity_stack = intensity_stack[valid]
led_positions_used = led_positions[valid]
frame_scales = frame_scales[valid]


# Normalize source brightness.
#
# This is a FIRST-PASS approximation.
# We are correcting scalar LED brightness only,
# not spatial illumination nonuniformity.
for i in range(len(intensity_stack)):

    intensity_stack[i] /= frame_scales[i]


# ============================================================
# 7. SHOW CROPPED INPUTS
# ============================================================

n = len(intensity_stack)

cols = 5
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(14, 3 * rows)
)

axes = np.asarray(axes).reshape(-1)

for i, ax in enumerate(axes):

    if i >= n:
        ax.axis("off")
        continue

    r, c = led_positions_used[i]

    ax.imshow(
        intensity_stack[i],
        cmap="gray"
    )

    ax.set_title(
        f"r{int(r):02d} c{int(c):02d}"
    )

    ax.axis("off")


fig.suptitle(
    "FP input ROI - normalized Green G1"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "01_input_roi_montage.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 8. LED ANGLES
# ============================================================

illumination_na = []

for r, c in led_positions_used:

    x_led = (
        int(c) - CENTER_COL
    ) * LED_PITCH_MM

    y_led = (
        int(r) - CENTER_ROW
    ) * LED_PITCH_MM

    z_led = LED_DISTANCE_MM

    distance = np.sqrt(
        x_led**2
        + y_led**2
        + z_led**2
    )

    na_x = x_led / distance
    na_y = y_led / distance

    illumination_na.append(
        [na_x, na_y]
    )


illumination_na = np.array(
    illumination_na
)


# ============================================================
# 9. FIRST-PASS NORMALIZED FOURIER MODEL
#
# Important:
# We use an empirical discrete Fourier geometry here.
# This is for checking whether the dataset reconstructs at all.
# ============================================================

Ny, Nx = intensity_stack.shape[1:]

# High-resolution reconstruction grid.
#
# Start modestly so the first test runs quickly.
UPSAMPLE = 3

Ny_hr = Ny * UPSAMPLE
Nx_hr = Nx * UPSAMPLE


# ------------------------------------------------------------
# Effective pupil radius
#
# Because physical NA is not yet calibrated accurately,
# convert EFFECTIVE_NA into a normalized pupil size.
#
# Clamp to a reasonable range for this first diagnostic run.
# ------------------------------------------------------------

pupil_radius = int(
    0.32 * min(Nx, Ny)
)

print()
print(
    "First-pass pupil radius:",
    pupil_radius,
    "Fourier pixels"
)


# Exact-size Fourier coordinate grid
y_coords = np.arange(Ny) - Ny // 2
x_coords = np.arange(Nx) - Nx // 2

xx, yy = np.meshgrid(
    x_coords,
    y_coords
)

pupil = (
    xx**2 + yy**2
    <= pupil_radius**2
).astype(np.float32)

print("Low-resolution patch shape:", (Ny, Nx))
print("Pupil shape:", pupil.shape)


# ------------------------------------------------------------
# Convert LED geometry to spectrum shifts.
#
# For this first diagnostic reconstruction,
# scale physical angular positions into a Fourier grid
# with substantial overlap.
#
# Later, once reconstruction works, this should be replaced
# with physically calibrated k-space mapping.
# ------------------------------------------------------------

na_x = illumination_na[:, 0]
na_y = illumination_na[:, 1]

max_nax = np.max(
    np.abs(na_x)
)

max_nay = np.max(
    np.abs(na_y)
)

# Keep adjacent pupils overlapping.
max_shift_x = int(
    0.45 * Nx
)

max_shift_y = int(
    0.25 * Ny
)


if max_nax > 0:
    shifts_x = np.round(
        na_x / max_nax
        * max_shift_x
    ).astype(int)
else:
    shifts_x = np.zeros_like(
        na_x,
        dtype=int
    )


if max_nay > 0:
    shifts_y = np.round(
        na_y / max_nay
        * max_shift_y
    ).astype(int)
else:
    shifts_y = np.zeros_like(
        na_y,
        dtype=int
    )


print()
print("Fourier shifts:")

for i, (r, c) in enumerate(
    led_positions_used
):

    print(
        f"r{int(r):02d}_c{int(c):02d}: "
        f"dx={shifts_x[i]:+d}, "
        f"dy={shifts_y[i]:+d}"
    )


# ============================================================
# 10. INITIAL OBJECT ESTIMATE
# ============================================================

# Use the LED closest to optical axis.
dist_from_center = (
    illumination_na[:, 0]**2
    +
    illumination_na[:, 1]**2
)

center_index = np.argmin(
    dist_from_center
)

central_intensity = intensity_stack[
    center_index
]

central_amplitude = np.sqrt(
    central_intensity
)


# Upsample central amplitude as initial guess
initial_amp = cv2.resize(
    central_amplitude,
    (Nx_hr, Ny_hr),
    interpolation=cv2.INTER_CUBIC
)

object_field = initial_amp.astype(
    np.complex64
)

object_spectrum = np.fft.fftshift(
    np.fft.fft2(
        np.fft.ifftshift(
            object_field
        )
    )
).astype(np.complex64)


# ============================================================
# 11. FP ITERATIONS
# ============================================================

cy = Ny_hr // 2
cx = Nx_hr // 2

half_y = Ny // 2
half_x = Nx // 2


errors = []


for iteration in range(
    N_ITERATIONS
):

    iteration_error = 0.0

    for i in range(n):

        shift_x = shifts_x[i]
        shift_y = shifts_y[i]

        center_x = cx + shift_x
        center_y = cy + shift_y


        x0 = center_x - half_x
        y0 = center_y - half_y

        x1 = x0 + Nx
        y1 = y0 + Ny


        if (
            x0 < 0
            or y0 < 0
            or x1 > Nx_hr
            or y1 > Ny_hr
        ):

            print(
                "Skipping Fourier patch "
                "outside HR grid:",
                i
            )

            continue


        patch = object_spectrum[
            y0:y1,
            x0:x1
        ].copy()


        # Apply pupil
        lowres_spectrum = (
            patch
            *
            pupil
        )


        predicted_field = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(
                    lowres_spectrum
                )
            )
        )


        predicted_amp = np.abs(
            predicted_field
        )


        measured_amp = np.sqrt(
            intensity_stack[i]
        )


        # Amplitude constraint
        updated_field = (
            measured_amp
            *
            np.exp(
                1j
                *
                np.angle(
                    predicted_field
                )
            )
        )


        updated_spectrum = np.fft.fftshift(
            np.fft.fft2(
                np.fft.ifftshift(
                    updated_field
                )
            )
        )


        # Spectrum correction
        difference = (
            updated_spectrum
            -
            lowres_spectrum
        )


        # Binary pupil -> simple alternating projection update
        patch = (
            patch
            +
            pupil
            *
            difference
        )


        object_spectrum[
            y0:y1,
            x0:x1
        ] = patch


        # Normalized amplitude error
        denom = (
            np.mean(measured_amp)
            + 1e-8
        )

        iteration_error += (
            np.mean(
                np.abs(
                    measured_amp
                    -
                    predicted_amp
                )
            )
            /
            denom
        )


    iteration_error /= n

    errors.append(
        iteration_error
    )

    print(
        f"Iteration "
        f"{iteration+1:02d}/{N_ITERATIONS}: "
        f"error = {iteration_error:.5f}"
    )


# ============================================================
# 12. FINAL RECONSTRUCTION
# ============================================================

reconstructed_field = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(
            object_spectrum
        )
    )
)

reconstructed_amplitude = np.abs(
    reconstructed_field
)

reconstructed_intensity = (
    reconstructed_amplitude**2
)


# Normalize only for display
display_reconstruction = (
    reconstructed_amplitude
    /
    (
        np.percentile(
            reconstructed_amplitude,
            99.8
        )
        + 1e-8
    )
)

display_reconstruction = np.clip(
    display_reconstruction,
    0,
    1
)


# ============================================================
# 13. BASELINE VS FP
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 6)
)

axes[0].imshow(
    central_intensity,
    cmap="gray"
)

axes[0].set_title(
    "Central illumination baseline"
)

axes[0].axis("off")


axes[1].imshow(
    display_reconstruction,
    cmap="gray"
)

axes[1].set_title(
    "First-pass FP reconstructed amplitude"
)

axes[1].axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_baseline_vs_fp.png"
    ),
    dpi=250
)

plt.show()


# ============================================================
# 14. ERROR CURVE
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    np.arange(
        1,
        N_ITERATIONS + 1
    ),
    errors,
    marker="o"
)

plt.xlabel(
    "Iteration"
)

plt.ylabel(
    "Normalized amplitude error"
)

plt.title(
    "FP convergence"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_convergence.png"
    ),
    dpi=200
)

plt.show()


# ============================================================
# 15. SAVE RESULTS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "fp_amplitude.npy"
    ),
    reconstructed_amplitude
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "fp_intensity.npy"
    ),
    reconstructed_intensity
)


print()
print("==========================================")
print("FIRST-PASS FP COMPLETE")
print("==========================================")

print(
    "Reconstruction shape:",
    reconstructed_amplitude.shape
)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)
