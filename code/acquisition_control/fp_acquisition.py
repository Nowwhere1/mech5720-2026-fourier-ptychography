import argparse
import time

from led_driver import HT16K33Matrix


def parse_indices(values, name):
    result = [int(v) for v in values]

    for v in result:
        if not 0 <= v <= 7:
            raise ValueError(f"{name} values must be in 0..7")

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Manual single-LED scan for Fourier ptychography acquisition. "
            "The script lights one LED at a time and waits for the user "
            "to capture a DNG image."
        )
    )

    parser.add_argument(
        "--rows",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Rows to scan, e.g. --rows 2 3 4 5 6"
    )

    parser.add_argument(
        "--cols",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5", "6", "7"],
        help="Columns to scan, e.g. --cols 2 3 4 5 6"
    )

    parser.add_argument(
        "--brightness",
        type=int,
        default=15,
        help="HT16K33 brightness, 0..15 (default: 15)"
    )

    parser.add_argument(
        "--off-delay",
        type=float,
        default=0.5,
        help="Delay after clearing LEDs, in seconds (default: 0.5)"
    )

    parser.add_argument(
        "--on-delay",
        type=float,
        default=1.0,
        help="Settling delay after turning LED on, in seconds (default: 1.0)"
    )

    args = parser.parse_args()

    rows = parse_indices(args.rows, "row")
    cols = parse_indices(args.cols, "col")

    positions = [(r, c) for r in rows for c in cols]

    matrix = HT16K33Matrix(brightness=args.brightness)

    print()
    print("MECH5720 FP manual LED acquisition")
    print("---------------------------------")
    print(f"Number of LED positions: {len(positions)}")
    print("Suggested image filename format: r##_c##.dng")
    print()

    try:
        for index, (row, col) in enumerate(positions, start=1):
            matrix.clear()
            time.sleep(args.off_delay)

            matrix.show_only(row, col)
            time.sleep(args.on_delay)

            print(
                f"[{index:02d}/{len(positions):02d}] "
                f"LED r{row:02d}_c{col:02d} ON"
            )

            response = input(
                "Capture DNG, then press ENTER to continue "
                "(or type q to stop): "
            ).strip().lower()

            if response == "q":
                print("Acquisition stopped by user.")
                break

        print("Scan complete.")

    except KeyboardInterrupt:
        print("\nAcquisition interrupted.")

    finally:
        matrix.close()


if __name__ == "__main__":
    main()
