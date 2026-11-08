# LED control scripts

These scripts reproduce the LED-control workflow used for the MECH5720
Fourier ptychography setup.

Hardware assumptions:
- Raspberry Pi I2C bus 1
- HT16K33 LED controller
- I2C address: `0x70`
- 8x8 red LED matrix
- Physical row mapping: `[6, 5, 4, 3, 2, 1, 0, 7]`

Install dependency:

```bash
pip install smbus2
```

Enable I2C on the Raspberry Pi before running the scripts.

## Files

- `led_driver.py` — common HT16K33 driver
- `led_hold.py` — turn on one LED and hold it
- `led_live.py` — central 2x2 or 4x4 pattern for alignment/focus
- `fp_acquisition.py` — manual sequential single-LED scan

## Examples

Single LED:

```bash
python led_hold.py 3 3
```

Central 2x2 alignment pattern:

```bash
python led_live.py --size 2
```

Central 4x4 alignment pattern:

```bash
python led_live.py --size 4
```

Full 8x8 manual scan:

```bash
python fp_acquisition.py
```

Central 5x5 scan:

```bash
python fp_acquisition.py --rows 2 3 4 5 6 --cols 2 3 4 5 6
```

For the report dataset, only one LED should be active per exposure during
angular acquisition. The alignment pattern is for setup/focusing only.
