"""
Entry point for running utp as a module: python -m utp
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
