import argparse
import time

from led_driver import HT16K33Matrix


def main():
    parser = argparse.ArgumentParser(
        description="Hold one LED on until Ctrl+C."
    )
    parser.add_argument("row", type=int, help="LED row, 0..7")
    parser.add_argument("col", type=int, help="LED column, 0..7")
    parser.add_argument(
        "--brightness",
        type=int,
        default=15,
        help="HT16K33 brightness, 0..15 (default: 15)"
    )
    args = parser.parse_args()

    matrix = HT16K33Matrix(brightness=args.brightness)

    try:
        matrix.show_only(args.row, args.col)
        print(
            f"LED r{args.row}_c{args.col} is ON. "
            "Press Ctrl+C to turn it off."
        )

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping.")

    finally:
        matrix.close()


if __name__ == "__main__":
    main()
