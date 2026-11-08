from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "fp_reconstruction"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STACK_PATH = DATA_DIR / "corrected_stack.npy"
LED_PATH = DATA_DIR / "led_positions.npy"
ROI_PATH = DATA_DIR / "roi.json"

WAVELENGTH_NM = 630.0
LED_PITCH_MM = 4.0
LED_DISTANCE_MM = 80.0
CENTER_ROW = 4
CENTER_COL = 4

# First-pass diagnostic parameter only; not a calibrated pupil NA.
EFFECTIVE_NA = 0.067

N_ITERATIONS = 10
UPSAMPLE = 3

for path in (STACK_PATH, LED_PATH, ROI_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Expected repository data under data/fp_reconstruction/."
        )

stack = np.load(STACK_PATH).astype(np.float32)
led_positions = np.load(LED_PATH)

with open(ROI_PATH, "r", encoding="utf-8") as f:
    roi = json.load(f)

x = int(roi["x"])
y = int(roi["y"])
w = int(roi["width"])
h = int(roi["height"])

print("Input stack:", stack.shape)
print("LED positions:", led_positions.shape)
print(f"ROI: x={x}, y={y}, width={w}, height={h}")

n_images, H, W = stack.shape

if x < 0 or y < 0 or x + w > W or y + h > H:
    raise ValueError(
        "Saved ROI falls outside the corrected stack. "
        f"Stack size is {W}x{H}; ROI is x={x}, y={y}, w={w}, h={h}."
    )

roi_stack = stack[:, y:y+h, x:x+w].copy()
intensity_stack = np.maximum(roi_stack, 0.0)

frame_scales = np.array(
    [np.percentile(img, 99.5) for img in intensity_stack],
    dtype=np.float32
)

valid = frame_scales > 1.0
intensity_stack = intensity_stack[valid]
led_positions_used = led_positions[valid]
frame_scales = frame_scales[valid]

for i in range(len(intensity_stack)):
    intensity_stack[i] /= frame_scales[i]

n = len(intensity_stack)
print(f"Frames retained: {n} / {len(valid)}")

illumination_na = []
for r, c in led_positions_used:
    x_led = (int(c) - CENTER_COL) * LED_PITCH_MM
    y_led = (int(r) - CENTER_ROW) * LED_PITCH_MM
    z_led = LED_DISTANCE_MM

    distance = np.sqrt(x_led**2 + y_led**2 + z_led**2)
    illumination_na.append([x_led / distance, y_led / distance])

illumination_na = np.asarray(illumination_na, dtype=np.float32)

Ny, Nx = intensity_stack.shape[1:]
Ny_hr = Ny * UPSAMPLE
Nx_hr = Nx * UPSAMPLE

# Empirical first-pass Fourier model used for the diagnostic reconstruction.
pupil_radius = int(0.32 * min(Nx, Ny))

y_coords = np.arange(Ny) - Ny // 2
x_coords = np.arange(Nx) - Nx // 2
xx, yy = np.meshgrid(x_coords, y_coords)

pupil = ((xx**2 + yy**2) <= pupil_radius**2).astype(np.float32)

na_x = illumination_na[:, 0]
na_y = illumination_na[:, 1]

max_nax = np.max(np.abs(na_x))
max_nay = np.max(np.abs(na_y))

max_shift_x = int(0.45 * Nx)
max_shift_y = int(0.25 * Ny)

if max_nax > 0:
    shifts_x = np.round(na_x / max_nax * max_shift_x).astype(int)
else:
    shifts_x = np.zeros_like(na_x, dtype=int)

if max_nay > 0:
    shifts_y = np.round(na_y / max_nay * max_shift_y).astype(int)
else:
    shifts_y = np.zeros_like(na_y, dtype=int)

dist_from_center = illumination_na[:, 0]**2 + illumination_na[:, 1]**2
center_index = int(np.argmin(dist_from_center))

central_intensity = intensity_stack[center_index]
central_amplitude = np.sqrt(central_intensity)

initial_amp = cv2.resize(
    central_amplitude,
    (Nx_hr, Ny_hr),
    interpolation=cv2.INTER_CUBIC
)

object_field = initial_amp.astype(np.complex64)
object_spectrum = np.fft.fftshift(
    np.fft.fft2(np.fft.ifftshift(object_field))
).astype(np.complex64)

cy = Ny_hr // 2
cx = Nx_hr // 2
half_y = Ny // 2
half_x = Nx // 2

errors = []

for iteration in range(N_ITERATIONS):
    iteration_error = 0.0
    used_this_iteration = 0

    for i in range(n):
        shift_x = int(shifts_x[i])
        shift_y = int(shifts_y[i])

        center_x = cx + shift_x
        center_y = cy + shift_y

        x0 = center_x - half_x
        y0 = center_y - half_y
        x1 = x0 + Nx
        y1 = y0 + Ny

        if x0 < 0 or y0 < 0 or x1 > Nx_hr or y1 > Ny_hr:
            continue

        patch = object_spectrum[y0:y1, x0:x1].copy()
        lowres_spectrum = patch * pupil

        predicted_field = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(lowres_spectrum))
        )

        predicted_amp = np.abs(predicted_field)
        measured_amp = np.sqrt(intensity_stack[i])

        updated_field = measured_amp * np.exp(
            1j * np.angle(predicted_field)
        )

        updated_spectrum = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(updated_field))
        )

        difference = updated_spectrum - lowres_spectrum
        patch = patch + pupil * difference
        object_spectrum[y0:y1, x0:x1] = patch

        denom = np.mean(measured_amp) + 1e-8
        iteration_error += (
            np.mean(np.abs(measured_amp - predicted_amp)) / denom
        )
        used_this_iteration += 1

    if used_this_iteration == 0:
        raise RuntimeError("No valid Fourier patches were used.")

    iteration_error /= used_this_iteration
    errors.append(iteration_error)

    print(
        f"Iteration {iteration + 1:02d}/{N_ITERATIONS}: "
        f"error = {iteration_error:.5f}"
    )

reconstructed_field = np.fft.fftshift(
    np.fft.ifft2(np.fft.ifftshift(object_spectrum))
)

reconstructed_amplitude = np.abs(reconstructed_field)
reconstructed_intensity = reconstructed_amplitude**2

display_reconstruction = reconstructed_amplitude / (
    np.percentile(reconstructed_amplitude, 99.8) + 1e-8
)
display_reconstruction = np.clip(display_reconstruction, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

axes[0].imshow(central_intensity, cmap="gray")
axes[0].set_title("Central illumination baseline")
axes[0].axis("off")

axes[1].imshow(display_reconstruction, cmap="gray")
axes[1].set_title("First-pass FP reconstructed amplitude")
axes[1].axis("off")

plt.tight_layout()
key_figure_path = RESULTS_DIR / "baseline_vs_fp.png"
plt.savefig(key_figure_path, dpi=250, bbox_inches="tight")
plt.close(fig)

fig = plt.figure(figsize=(7, 5))
plt.plot(np.arange(1, N_ITERATIONS + 1), errors, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Normalized amplitude error")
plt.title("FP convergence")
plt.grid()
plt.tight_layout()

convergence_path = RESULTS_DIR / "convergence.png"
plt.savefig(convergence_path, dpi=200, bbox_inches="tight")
plt.close(fig)

np.save(RESULTS_DIR / "fp_amplitude.npy", reconstructed_amplitude)
np.save(RESULTS_DIR / "fp_intensity.npy", reconstructed_intensity)

print()
print("==========================================")
print("FIRST-PASS FP REPRODUCTION COMPLETE")
print("==========================================")
print("Key figure:", key_figure_path)
print("Convergence figure:", convergence_path)
