# Frozen builds: run from the exe's folder so Numba's kernel cache lands in one stable place.
import os
import sys

if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))
