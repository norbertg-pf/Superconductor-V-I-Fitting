"""Allow ``python -m src.fitting`` to launch the standalone window."""

from .standalone import main

if __name__ == "__main__":
    raise SystemExit(main())
