"""Backward-compatible Windows entry point for the cross-platform setup check."""
from check_setup import main

if __name__ == "__main__":
    raise SystemExit(main())
