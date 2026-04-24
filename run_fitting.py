"""Entry point used by PyInstaller for standalone executable builds."""

from fitting.standalone import main

if __name__ == "__main__":
    raise SystemExit(main())
