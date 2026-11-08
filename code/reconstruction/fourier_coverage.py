import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

LED_PITCH_MM = 4.0
LED_DISTANCE_MM = 80.0

CENTER_ROW = 4
CENTER_COL = 4

WAVELENGTH_NM = 630.0

# First estimate only
NA_OBJECTIVE = 0.067

LED_POSITIONS = [
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
]


# ============================================================
# CALCULATE ILLUMINATION NA
# ============================================================

led_data = []

for row, col in LED_POSITIONS:

    x = (col - CENTER_COL) * LED_PITCH_MM
    y = (row - CENTER_ROW) * LED_PITCH_MM
    z = LED_DISTANCE_MM

    distance = np.sqrt(x**2 + y**2 + z**2)

    na_x = x / distance
    na_y = y / distance

    na_total = np.sqrt(
        na_x**2 + na_y**2
    )

    theta_deg = np.degrees(
        np.arcsin(na_total)
    )

    led_data.append(
        (row, col, na_x, na_y, na_total, theta_deg)
    )


# ============================================================
# PRINT LED INFORMATION
# ============================================================

print()
print("LED illumination geometry")
print("=" * 65)

print(
    "LED       NA_x       NA_y       NA_total    angle [deg]"
)

for row, col, na_x, na_y, na_total, theta in led_data:

    print(
        f"r{row:02d}_c{col:02d}   "
        f"{na_x:+.4f}    "
        f"{na_y:+.4f}    "
        f"{na_total:.4f}       "
        f"{theta:.3f}"
    )


max_illum_na = max(
    d[4] for d in led_data
)

synthetic_na = NA_OBJECTIVE + max_illum_na

print()
print("=" * 65)
print(f"Objective NA estimate = {NA_OBJECTIVE:.4f}")
print(f"Maximum illumination NA = {max_illum_na:.4f}")
print(f"Approx. synthetic NA = {synthetic_na:.4f}")
print("=" * 65)


# ============================================================
# PLOT FOURIER / NA COVERAGE
# ============================================================

fig, ax = plt.subplots(figsize=(8, 7))

angle = np.linspace(
    0,
    2 * np.pi,
    300
)

for row, col, na_x, na_y, na_total, theta in led_data:

    # Each LED shifts the object's Fourier spectrum.
    # Each circle represents the pupil support for that illumination.
    circle_x = na_x + NA_OBJECTIVE * np.cos(angle)
    circle_y = na_y + NA_OBJECTIVE * np.sin(angle)

    ax.plot(
        circle_x,
        circle_y,
        linewidth=1
    )

    ax.plot(
        na_x,
        na_y,
        "o",
        markersize=4
    )

    ax.text(
        na_x + 0.003,
        na_y + 0.003,
        f"r{row}c{col}",
        fontsize=7
    )


# Optical axis
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)

ax.set_xlabel("NA x")
ax.set_ylabel("NA y")

ax.set_title(
    "Estimated Fourier Coverage\n"
    f"Objective NA = {NA_OBJECTIVE:.3f}"
)

ax.set_aspect("equal")
ax.grid()

plt.show()