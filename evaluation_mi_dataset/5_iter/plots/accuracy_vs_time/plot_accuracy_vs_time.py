#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from plot_suite import generate  # noqa: E402

generate('accuracy_vs_time', SCRIPT_DIR)
