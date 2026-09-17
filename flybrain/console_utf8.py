"""Import this first in any script that prints Hebrew, so it works on
Windows consoles too (which often default to cp1252/cp1255, not UTF-8)."""

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
