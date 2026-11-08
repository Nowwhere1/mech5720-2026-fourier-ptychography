from smbus2 import SMBus
import time


class HT16K33Matrix:
    """
    Minimal driver for the 8x8 red LED matrix used in the MECH5720 FP setup.

    Hardware assumptions used in the original setup:
    - HT16K33 I2C controller
    - I2C address 0x70
    - Raspberry Pi SMBus / I2C
    - Physical row-to-bit mapping: [6, 5, 4, 3, 2, 1, 0, 7]
    """

    ROW_TO_BIT = [6, 5, 4, 3, 2, 1, 0, 7]

    def __init__(self, bus_number=1, address=0x70, brightness=15):
        self.address = address
        self.bus = SMBus(bus_number)
        self.buffer = [0x00] * 16

        # HT16K33 oscillator on
        self.bus.write_byte(self.address, 0x21)

        # Display on, blink off
        self.bus.write_byte(self.address, 0x81)

        # Brightness: 0-15
        self.set_brightness(brightness)

        self.clear()

    def set_brightness(self, brightness):
        brightness = max(0, min(15, int(brightness)))
        self.bus.write_byte(self.address, 0xE0 | brightness)

    def _validate(self, row, col):
        if not (0 <= row <= 7 and 0 <= col <= 7):
            raise ValueError("row and col must each be in the range 0..7")

    def _write(self):
        self.bus.write_i2c_block_data(
            self.address,
            0x00,
            self.buffer
        )

    def clear(self):
        self.buffer = [0x00] * 16
        self._write()

    def set_led(self, row, col, on=True, update=True):
        """
        Turn one physical LED on or off.

        Coordinates are zero-based:
            row = 0..7
            col = 0..7
        """
        self._validate(row, col)

        byte_index = col * 2
        bit = self.ROW_TO_BIT[row]
        mask = 1 << bit

        if on:
            self.buffer[byte_index] |= mask
        else:
            self.buffer[byte_index] &= ~mask

        if update:
            self._write()

    def show_only(self, row, col):
        """Turn on exactly one LED and turn all others off."""
        self.buffer = [0x00] * 16
        self.set_led(row, col, on=True, update=False)
        self._write()

    def show_pattern(self, positions):
        """Display a list of (row, col) LED coordinates."""
        self.buffer = [0x00] * 16
        for row, col in positions:
            self.set_led(row, col, on=True, update=False)
        self._write()

    def close(self):
        try:
            self.clear()
            time.sleep(0.05)
        finally:
            self.bus.close()
