#!/usr/bin/env python3
"""MECH5720 Group 11: original DNG files -> GREEN G1 -> paper Figure 3 PNG.

This is the FULL raw-DNG processing pipeline, not CSV-to-plot.

Easiest in Python IDLE (F5):
    Put this script next to BOTH original iris_1mm.zip and iris_4mm.zip,
    OR next to extracted folders named iris_1mm and iris_4mm.
    Then press F5 to run it. ZIPs do NOT need to be extracted.

Alternatively run from PowerShell with explicit folders OR ZIPs:
    python Figure3_From_DNG_Green_G1.py --iris1 "D:/data/iris_1mm" --iris4 "D:/data/iris_4mm"
    python Figure3_From_DNG_Green_G1.py --iris1 iris_1mm.zip --iris4 iris_4mm.zip

Original data must contain target_200um, target_250um, target_300um
subfolders and reference-frame subfolders (1 mm: dark_frame; 4 mm: ambient).


Dependencies:
    python -m pip install numpy scipy matplotlib tifffile

Outputs (next to script by default, or under --outdir):
    illumination_centroids.csv  (one measurement per raw DNG image)
    Figure3_G1_centre_displacement_points.csv
    Figure3_G1_centre_displacement_two_iris.png
    Figure3_G1_centre_displacement_two_iris.pdf
    Figure3_G1_centre_displacement_two_iris.svg

The measured quantity is image-plane bright-illumination-region displacement,
NOT resolution improvement, sensor movement, or Fourier-frequency overlap.
The plotted whiskers are the min/max across THREE target scans, not independent
repeat-exposure uncertainty. Both irises have different illumination settings.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import zipfile
from pathlib import Path
from contextlib import contextmanager

import numpy as np
from scipy.ndimage import gaussian_filter, label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_PATTERN = re.compile(r"iris(\d+)mm_(\d+)um_r(\d+)_c(\d+)\.dng$", re.I)
THRESHOLDS = (0.30, 0.40, 0.45, 0.50, 0.60)


@contextmanager
def open_dng_collection(source_path: Path):
    """Read .dng from either the original .zip OR an extracted directory tree."""
    source_path = source_path.expanduser().resolve()
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".dng")]
            yield members, archive.read
    elif source_path.is_dir():
        members = sorted(p.relative_to(source_path).as_posix()
                         for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() == ".dng")
        def read_member(name):
            return (source_path / Path(name)).read_bytes()
        yield members, read_member
    else:
        raise FileNotFoundError(f"Input is not a .zip archive or DNG directory: {source_path}")


def is_in_folder(name: str, folder: str) -> bool:
    return folder.casefold() in (p.casefold() for p in name.replace("\\", "/").split("/"))


def dng_green_g1(read_member, member: str) -> np.ndarray:
    """Read raw 10-bit packed Bayer and return only G1, at even rows/even cols.

    We explicitly check CFA tags rather than demosaicing/interpolating.
    This matches the original MECH5720 preprocessing pipeline.
    """
    import tifffile  # only needed for the raw-DNG pathway
    data = read_member(member)
    with tifffile.TiffFile(io.BytesIO(data)) as tiff:
        page = tiff.pages[0]
        height, width = page.shape
        bits = int(page.tags["BitsPerSample"].value)
        compression = int(page.tags["Compression"].value)
        cfa = bytes(page.tags["CFAPattern"].value)
        offsets = page.tags["StripOffsets"].value
        counts = page.tags["StripByteCounts"].value
        off = int(offsets[0] if isinstance(offsets, (tuple, list)) else offsets)
        size = int(counts[0] if isinstance(counts, (tuple, list)) else counts)

    if bits != 10 or compression != 1 or cfa != bytes([1, 2, 0, 1]):
        raise ValueError(f"Unexpected packed DNG format in {member}: "
                         f"bits={bits}, compression={compression}, CFA={list(cfa)}")
    if height % 2 or width % 4:
        raise ValueError(f"Unexpected dimensions ({height}, {width}) in {member}")
    packed = np.frombuffer(data, dtype=np.uint8, count=size, offset=off)
    if packed.size != height * width * 10 // 8:
        raise ValueError(f"Unexpected DNG sample byte count in {member}")
    even_rows = packed.reshape(height, width // 4, 5)[0::2].astype(np.uint16)
    g1 = np.empty((height // 2, width // 2), dtype=np.uint16)
    g1[:, 0::2] = (even_rows[:, :, 0] << 2) | (even_rows[:, :, 1] >> 6)
    g1[:, 1::2] = ((even_rows[:, :, 2] & 15) << 6) | (even_rows[:, :, 3] >> 2)
    return g1


def region_centroid(image: np.ndarray, fraction: float) -> dict | None:
    """Segment the dominant bright region and compute its intensity-weighted centre."""
    smooth = gaussian_filter(image.astype(np.float32), sigma=7, mode="reflect")
    cutoff = max(1.2, fraction * float(smooth.max()))
    labels, count = label(smooth > cutoff)
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(np.argmax(sizes))
    if int(sizes[selected]) < 40:
        return None
    yy, xx = np.where(labels == selected)
    weights = smooth[yy, xx]
    return dict(
        x=float(np.average(xx, weights=weights)),
        y=float(np.average(yy, weights=weights)),
        clipped=bool(xx.min() <= 1 or yy.min() <= 1 or
                     xx.max() >= image.shape[1] - 2 or yy.max() >= image.shape[0] - 2),
        peak_DN=float(smooth.max()),
        region_area_px2=int(sizes[selected]),
    )


def measure_dng_source(source_path: Path, iris: int) -> list[dict]:
    """Compute centroids directly from raw DNG files (ZIP or extracted directory)."""
    records = []
    with open_dng_collection(source_path) as (members, read_member):
        ref_type = "dark_frame" if iris == 1 else "ambient"
        ref_files = sorted(n for n in members if is_in_folder(n, ref_type))
        if not ref_files:
            raise FileNotFoundError(
                f"No reference DNG files found in '{ref_type}' under {source_path}. "
                "Keep the original reference frames alongside target folders.")
        reference = np.median(np.stack([dng_green_g1(read_member, fn) for fn in ref_files]),
                              axis=0).astype(np.float32)
        print(f"{iris} mm: subtracting median of {len(ref_files)} {ref_type} DNG frames", flush=True)
        for target in (200, 250, 300):
            folder = f"target_{target}um"
            names = sorted((n for n in members if is_in_folder(n, folder)
                            and RAW_PATTERN.search(n)),
                           key=lambda name: tuple(map(int, RAW_PATTERN.search(name).groups()[2:])))
            required = 9 if iris == 1 else 25
            if len(names) != required:
                raise ValueError(f"Expected {required} raw DNG target images in {folder} "
                                 f"for iris {iris} mm, but found {len(names)} in {source_path}.")
            for name in names:
                m = RAW_PATTERN.search(name)
                row, col = int(m.group(3)), int(m.group(4))
                g1 = np.maximum(dng_green_g1(read_member, name).astype(np.float32) - reference,
                                0).astype(np.uint16)
                base = region_centroid(g1, 0.45)
                if base is None:
                    raise ValueError(f"No measurable illumination region in DNG image {name}")
                alternate = [region_centroid(g1, t) for t in THRESHOLDS]
                alternate = [item for item in alternate if item is not None]
                xs = [v["x"] for v in alternate]
                ys = [v["y"] for v in alternate]
                records.append(dict(iris_mm=iris, target_um_nominal=target,
                                    led_row=row, led_col=col, **base,
                                    x_threshold_range_px=float(max(xs) - min(xs)),
                                    y_threshold_range_px=float(max(ys) - min(ys))))
            print(f"  {target} um target: {len(names)} raw DNGs -> G1 centroids", flush=True)
    return records


def write_centroids(path: Path, rows: list[dict]) -> None:
    columns = ["iris_mm", "target_um_nominal", "led_row", "led_col", "x", "y", "clipped",
               "peak_DN", "region_area_px2", "x_threshold_range_px", "y_threshold_range_px"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict], outdir: Path) -> None:
    indexed = {(r["iris_mm"], r["target_um_nominal"], r["led_row"], r["led_col"]): r for r in rows}
    points = []
    for iris in (4, 1):
        mid_row, mid_col = (4, 4) if iris == 4 else (4, 2)
        for target in (200, 250, 300):
            centre = indexed[(iris, target, mid_row, mid_col)]
            for axis in ("horizontal", "vertical"):
                for step in (range(-2, 3) if iris == 4 else range(-1, 2)):
                    key = ((iris, target, mid_row + step, mid_col) if axis == "horizontal"
                           else (iris, target, mid_row, mid_col + step))
                    r = indexed[key]
                    if r["clipped"]:
                        continue
                    amount = (r["x"] - centre["x"]) if axis == "horizontal" else (centre["y"] - r["y"])
                    points.append(dict(iris_mm=iris, target_um=target, axis=axis,
                                       led_offset=step, centroid_shift_G1_px=amount))

    with (outdir / "Figure3_G1_centre_displacement_points.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), sharey=True)
    for ax, axis in zip(axes, ("horizontal", "vertical")):
        for iris in (4, 1):
            relevant = [p for p in points if p["axis"] == axis and p["iris_mm"] == iris]
            steps = sorted(set(p["led_offset"] for p in relevant))
            median = np.array([np.median([p["centroid_shift_G1_px"] for p in relevant
                                           if p["led_offset"] == s]) for s in steps])
            lower = np.array([min(p["centroid_shift_G1_px"] for p in relevant
                                  if p["led_offset"] == s) for s in steps])
            upper = np.array([max(p["centroid_shift_G1_px"] for p in relevant
                                  if p["led_offset"] == s) for s in steps])
            ax.errorbar(steps, median, yerr=np.stack([median - lower, upper - median]),
                        marker="o" if iris == 4 else "s", linestyle="-" if iris == 4 else "--",
                        linewidth=1.55, markersize=4.4, capsize=2.6, elinewidth=1.05,
                        label=f"{iris} mm iris", alpha=.95)
        ax.axhline(0, color="black", lw=.55, alpha=.35)
        ax.axvline(0, color="black", lw=.55, alpha=.35)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlim(-2.3, 2.3)
        ax.set_ylim(-258, 262)
        ax.grid(alpha=.18)
        ax.set_xlabel("LED offset from scan centre (steps)")
        ax.set_title("(a) LED row → horizontal shift" if axis == "horizontal"
                     else "(b) LED column → upward shift", pad=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Illumination-centroid shift (G1 pixels)")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].text(.025, .03, "4 mm outer column omitted (clipped)", fontsize=7.1,
                 transform=axes[1].transAxes, color="0.35")
    fig.suptitle("Image-plane bright-region displacement under single-LED switching",
                 fontsize=11.5, y=.985)
    fig.subplots_adjust(left=.09, right=.98, bottom=.22, top=.80, wspace=.16)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(outdir / f"Figure3_G1_centre_displacement_two_iris.{ext}", dpi=260)
    plt.close(fig)

    print(f"Created Figure 3 from {len(rows)} G1 frames; {len(points)} target-transect points")
    for iris in (1, 4):
        valid = [r for r in rows if r["iris_mm"] == iris and not r["clipped"]]
        spans = [np.hypot(r["x_threshold_range_px"], r["y_threshold_range_px"])
                 for r in valid if np.isfinite(r["x_threshold_range_px"])
                 and np.isfinite(r["y_threshold_range_px"])]
        if spans:
            print(f"{iris} mm: median centroid threshold span (30%-60%) = {np.median(spans):.1f} G1 px "
                  f"(NOT repeated-exposure uncertainty, n={len(spans)})")


def locate_source(name: str, script_dir: Path) -> Path | None:
    # Try current work directory and the script's own directory (IDLE can use either).
    for parent in (script_dir, Path.cwd()):
        path = parent / name
        if path.is_dir() or (path.is_file() and path.suffix.lower() == ".zip"):
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript Figure 3 PNG directly from Green G1 DNGs.")
    parser.add_argument("--iris1", type=Path, help="Original 1 mm DNG folder OR .zip archive")
    parser.add_argument("--iris4", type=Path, help="Original 4 mm DNG folder OR .zip archive")
    parser.add_argument("--outdir", type=Path, help="Output folder (default: folder containing this script)")
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    outdir = args.outdir or script_dir
    outdir.mkdir(parents=True, exist_ok=True)

    if bool(args.iris1) != bool(args.iris4):
        parser.error("Please supply --iris1 AND --iris4 together.")

    iris1 = (args.iris1 or locate_source("iris_1mm", script_dir)
             or locate_source("iris_1mm.zip", script_dir)
             or locate_source("iris_1mm(2).zip", script_dir))
    iris4 = (args.iris4 or locate_source("iris_4mm", script_dir)
             or locate_source("iris_4mm.zip", script_dir)
             or locate_source("iris_4mm(1).zip", script_dir))
    if not iris1 or not iris4:
        parser.error("Cannot find BOTH original DNG collections. Put iris_1mm.zip and "
                     "iris_4mm.zip beside this script (or extracted iris_1mm and iris_4mm folders), "
                     "then F5 in IDLE. Alternatively specify --iris1 and --iris4 paths.")

    print("Reading original DNG frames for Figure 3 (Green G1, no demosaicing)", flush=True)
    print(f"1 mm source: {iris1}", flush=True)
    print(f"4 mm source: {iris4}", flush=True)
    rows = measure_dng_source(iris1, 1) + measure_dng_source(iris4, 4)
    write_centroids(outdir / "illumination_centroids.csv", rows)
    make_figure(rows, outdir)
    print(f"\nSUCCESS: PNG generated at: {outdir / 'Figure3_G1_centre_displacement_two_iris.png'}")
    print("CSV (intermediate measurements) also saved for reproducibility.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
