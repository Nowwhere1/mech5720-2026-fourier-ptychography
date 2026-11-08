import argparse
import time

from led_driver import HT16K33Matrix


def centered_square(size):
    """
    Return a centered 2x2 or 4x4 LED pattern on an 8x8 matrix.
    """
    if size == 2:
        indices = [3, 4]
    elif size == 4:
        indices = [2, 3, 4, 5]
    else:
        raise ValueError("size must be 2 or 4")

    return [(r, c) for r in indices for c in indices]


def main():
    parser = argparse.ArgumentParser(
        description="Keep a central LED pattern on for alignment/focusing."
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[2, 4],
        default=2,
        help="Central square size: 2 or 4 (default: 2)"
    )
    parser.add_argument(
        "--brightness",
        type=int,
        default=15,
        help="HT16K33 brightness, 0..15 (default: 15)"
    )
    args = parser.parse_args()

    matrix = HT16K33Matrix(brightness=args.brightness)
    pattern = centered_square(args.size)

    try:
        matrix.show_pattern(pattern)
        print(
            f"Central {args.size}x{args.size} pattern is ON. "
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
