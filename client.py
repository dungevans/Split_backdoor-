import sys


def main():
    print("Split client mode is disabled. Use: python server.py --config config.yaml")
    return 1


if __name__ == "__main__":
    sys.exit(main())
