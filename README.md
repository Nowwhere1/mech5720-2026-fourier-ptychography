# MECH5720 2026 – Low-Cost Fourier Ptychographic Imaging

**Unit:** MECH5720 Imaging and Sensing  
**Year:** 2026  
**Group:** Group 11  
**Project:** Low-Cost Fourier Ptychographic Imaging with LED-Array Illumination: Implementation and Calibration Challenges

## Project Overview

This project investigates the implementation of a low-cost Fourier ptychographic (FP) imaging system using an LED matrix, a single imaging lens, and a Raspberry Pi camera.

In Fourier ptychography, a sequence of images is captured under different illumination angles. Each illumination direction shifts the accessible spatial-frequency region of the object. The measurements are then combined computationally to estimate a higher-resolution object.

A complete acquisition, preprocessing, calibration, and first-pass FP reconstruction pipeline was implemented in this project.

The computational pipeline successfully produced a reconstructed object estimate. However, the reconstruction contained strong spatial artefacts and did not demonstrate reliable resolution enhancement. Further experiments showed illumination-dependent image deformation and uncertainty in the effective pupil calibration, which limited agreement between the physical system and the simplified FP forward model.

## Experimental System

The optical path was approximately:

```text
LED matrix
    ↓
transmissive bar-pattern target
    ↓
single imaging lens
    ↓
Raspberry Pi camera
```

Main hardware:

| Component | Description |
|---|---|
| Camera | Raspberry Pi camera with OV5647 sensor |
| Imaging lens | Single convex lens, focal length approximately 30 mm |
| Illumination | 8 × 8 red LED matrix |
| LED controller | HT16K33 |
| LED pitch | Approximately 4 mm |
| Target | Printed transmissive bar-pattern target |
| Exposure time | 64000 µs |
| Nominal irises | 1 mm and 4 mm |

The printed target contained nominal bar widths of 300 µm, 250 µm, 200 µm, and 150 µm, with approximately equal opaque-bar and transparent-gap widths.

The 200 µm structures were reproduced substantially more reliably than the 150 µm structures, so no quantitative resolution claim below approximately 200 µm is made using the current printed target.

## Hardware Setup

The camera, imaging lens, and target were kept fixed during each angular acquisition sequence. Individual LEDs were activated sequentially to change the illumination direction.

The initial LED-to-target distance was approximately 80 mm. During the calibration investigation, this distance was progressively increased to approximately 380 mm.

The additional iris could not be placed at a well-defined pupil plane because of the physical lens and camera geometry. The nominal iris diameter should therefore not be interpreted directly as an accurately calibrated system numerical aperture.

## LED Control

The HT16K33 LED matrix was controlled using the Raspberry Pi I2C interface.

LED-control scripts are located in:

```text
code/acquisition/
```

Important scripts include:

```text
led_driver.py
led_hold.py
led_live.py
fp_acquisition.py
```

### Single LED

To turn on one LED:

```bash
python code/acquisition/led_hold.py 3 3
```

### Alignment Pattern

To display a central 2 × 2 LED pattern for alignment and focusing:

```bash
python code/acquisition/led_live.py --size 2
```

### Sequential Angular Acquisition

For a central 5 × 5 LED scan:

```bash
python code/acquisition/fp_acquisition.py --rows 2 3 4 5 6 --cols 2 3 4 5 6
```

Only one LED should be active during each FP measurement.

The logical LED coordinates should be checked against the physical matrix before acquisition because the HT16K33 memory layout does not necessarily correspond directly to the apparent physical row and column order.

## Image Acquisition

Raw measurements were captured in DNG format.

The main first-pass FP dataset used the nominal 4 mm iris. A 5 × 5 region of the LED matrix was initially captured. The central measurements provided the most consistent target illumination, and fifteen central LED positions were retained for the first-pass reconstruction.

A second experiment using the nominal 1 mm iris produced lower light throughput and a smaller usable angular region of approximately 3 × 3 LEDs.

An exposure time of 64000 µs was used because shorter exposures produced visible LED scanning/PWM banding.

## Preprocessing

The preprocessing pipeline was:

```text
DNG
 ↓
Bayer-channel extraction
 ↓
background correction
 ↓
fixed ROI
 ↓
intensity normalisation
```

The Green G1 Bayer channel was used for the main first-pass reconstruction.

Object-space pixel sampling was calibrated using the nominal 250 µm bar target. Since one black bar and one clear gap together formed an approximately 500 µm period, repeated measurements produced 28.16 µm/pixel and 28.32 µm/pixel. The mean value used for the analysis was approximately **28.24 µm/pixel**.

## Repository Structure

```text
mech5720-2026-fourier-ptychography/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── code/
│   ├── acquisition/
│   ├── preprocessing/
│   ├── reconstruction/
│   ├── calibration/
│   └── utils/
│
├── data/
│   └── fp_reconstruction/
│
├── reproduce/
│   └── reproduce_key_figure.py
│
└── results/
```

## Important Processing Scripts

| Script | Purpose |
|---|---|
| `preprocess_fp.py` | Preprocess raw FP measurements |
| `compare_bayer_channels.py` | Compare Bayer-channel responses |
| `measure_pixel_scale.py` | Measure object-space pixel calibration |
| `led_geometry.py` | Calculate LED illumination geometry |
| `fourier_coverage.py` | Visualise approximate Fourier coverage |
| `fp_reconstruct_firstpass.py` | Original first-pass FP reconstruction |
| `make_raw_led_montage.py` | Generate LED acquisition montage |
| `reproduce_key_figure.py` | Reproduce the main FP reconstruction figure |

## First-Pass Fourier Ptychographic Reconstruction

The reconstruction represents the object as a complex field:

```text
O = A exp(jφ)
```

The implementation in this repository is intentionally described as a **first-pass diagnostic reconstruction**.

The effective pupil and Fourier shifts were approximated using an empirical discrete model to determine whether the measured dataset could produce a stable FP reconstruction.

The effective pupil numerical aperture and illumination-to-Fourier-space mapping were not independently calibrated with sufficient accuracy for a fully quantitative FP reconstruction.

## Reproducing the Key Figure

The processed reconstruction dataset is located in:

```text
data/fp_reconstruction/
```

It contains:

```text
corrected_stack.npy
central_corrected_baseline.npy
led_positions.npy
metadata.json
roi.json
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then, from the repository root, run:

```bash
python reproduce/reproduce_key_figure.py
```

The script generates:

```text
results/baseline_vs_fp.png
results/convergence.png
```

## Main Result

The main comparison is stored at:

```text
results/baseline_vs_fp.png
```

The reconstruction algorithm successfully produced an object estimate and the numerical update converged over successive iterations.

However, the reconstructed amplitude contained strong wave-like and repeated spatial artefacts that were absent from the conventional central illumination image.

No reliable resolution enhancement was therefore demonstrated.

## Calibration Investigation

Several possible causes of the reconstruction failure were experimentally investigated.

| Test | Observation |
|---|---|
| Increased LED-to-target distance | Image deformation remained |
| Removed diffuser | No reliable improvement |
| Removed additional iris | Deformation remained |
| Repeated refocusing | No stable correction |
| Reduced LED brightness | Deformation remained |
| Per-LED flat-field correction | Illumination envelope reduced, deformation remained |

Under directional single-LED illumination, the parallel-bar target could appear narrowed or curved around the strongly illuminated region. The apparent geometry also changed with illumination position.

The effect is therefore described conservatively as **illumination-dependent image deformation** rather than being attributed to a single identified optical mechanism.

## Flat-Field Calibration

A per-LED flat-field experiment was used to test whether the deformation was primarily caused by spatially non-uniform illumination.

The correction was calculated approximately as:

```text
corrected = (target - dark) / (flat - dark + ε)
```

The flat-field correction reduced the large-scale illumination envelope, but did not remove the apparent deformation of the target bars.

The corresponding comparison is shown in:

```text
results/flat_field_comparison.png
```

## Known Limitations and Pitfalls

### Effective Pupil Uncertainty

The nominal 1 mm and 4 mm iris diameters do not correspond directly to a calibrated pupil numerical aperture because the iris could not be placed at a known pupil plane.

The nominal 4 mm aperture gives a first-order estimate of approximately:

```text
NA ≈ D / (2f) ≈ 4 / (2 × 30) ≈ 0.067
```

but this value is used only as a first-pass modelling parameter.

### Illumination Model

The simplified reconstruction assumes approximately plane-wave illumination and known angular shifts. The experimental illumination did not fully satisfy these assumptions across the selected field of view.

### Illumination-Dependent Image Deformation

Target geometry changed with illumination position. Possible contributors include optical aberration, residual defocus, imperfect plane-wave illumination, pupil uncertainty, transmissive-substrate effects, and limitations of the single-element lens.

The current experiments do not uniquely identify a dominant mechanism.

### Printed Target

The 150 µm structures were not reproduced reliably by the printed target. The 200 µm structures were substantially more consistent.

The current target therefore limits quantitative validation of resolution improvement.

## Full Raw Dataset

The GitHub repository contains the processed data required to reproduce the main reconstruction result and representative calibration results.

The complete raw DNG dataset is stored separately because of its larger file size.

**Full experimental dataset:**  
[https://drive.google.com/drive/folders/14qT0WaWSy1TUtSKhh6aftWu23FKJNJ-n?usp=sharing]

## Python Environment

Install dependencies using:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
matplotlib
opencv-python
rawpy
scipy
smbus2
```

`smbus2` is required only for Raspberry Pi / HT16K33 LED control.

## Results Included in This Repository

```text
results/
├── baseline_vs_fp.png
├── convergence.png
├── raw_15_LED_montage.png
├── corrected_15_LED_montage.png
├── bayer_channel_comparison.png
└── flat_field_comparison.png
```

## Authors

**Group 11**
MECH5720 Sensors and Signals 
The University of Sydney  
2026

## License

This project is released under the MIT License. See `LICENSE` for details.
