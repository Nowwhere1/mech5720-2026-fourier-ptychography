import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Geometry
# ============================================================

LED_PITCH_MM = 4.0
LED_TO_TARGET_MM = 80.0

CENTER_ROW = 4
CENTER_COL = 4

LED_POSITIONS = [
    (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
]


# ============================================================
# Calculate illumination geometry
# ============================================================

results = []

for row, col in LED_POSITIONS:

    # Physical offset relative to centre LED
    x_mm = (col - CENTER_COL) * LED_PITCH_MM
    y_mm = (row - CENTER_ROW) * LED_PITCH_MM

    z_mm = LED_TO_TARGET_MM

    # Illumination angles
    theta_x = np.degrees(np.arctan2(x_mm, z_mm))
    theta_y = np.degrees(np.arctan2(y_mm, z_mm))

    r_xy = np.sqrt(x_mm**2 + y_mm**2)

    theta_total = np.degrees(
        np.arctan2(r_xy, z_mm)
    )

    # Direction cosines / illumination NA components
    distance = np.sqrt(
        x_mm**2 + y_mm**2 + z_mm**2
    )

    na_x = x_mm / distance
    na_y = y_mm / distance

    na_total = np.sqrt(na_x**2 + na_y**2)

    results.append(
        (
            row,
            col,
            x_mm,
            y_mm,
            theta_x,
            theta_y,
            theta_total,
            na_x,
            na_y,
            na_total
        )
    )


# ============================================================
# Print results
# ============================================================

print()
print(
    "LED      x(mm)   y(mm)   "
    "theta_x   theta_y   theta_total   NA"
)

for r in results:

    row, col, x, y, tx, ty, tt, nax, nay, nat = r

    print(
        f"r{row:02d}_c{col:02d}  "
        f"{x:6.1f}  "
        f"{y:6.1f}  "
        f"{tx:8.3f}  "
        f"{ty:8.3f}  "
        f"{tt:11.3f}  "
        f"{nat:7.4f}"
    )


# ============================================================
# Plot illumination NA positions
# ============================================================

na_x_values = [r[7] for r in results]
na_y_values = [r[8] for r in results]

plt.figure(figsize=(7, 5))

plt.scatter(
    na_x_values,
    na_y_values,
    s=80
)

for r in results:

    row, col = r[0], r[1]
    na_x, na_y = r[7], r[8]

    plt.text(
        na_x + 0.002,
        na_y + 0.002,
        f"r{row}c{col}",
        fontsize=8
    )

plt.axhline(0, linewidth=0.8)
plt.axvline(0, linewidth=0.8)

plt.xlabel("Illumination NA x")
plt.ylabel("Illumination NA y")

plt.title(
    "Illumination angles of selected LEDs"
)

plt.axis("equal")
plt.grid()

plt.show()