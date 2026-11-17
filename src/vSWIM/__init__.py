#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:32:38 2026

@author: mrutala
"""

from importlib.metadata import version, PackageNotFoundError

import getData, vSWIM, backmap

try:
    __version__ = version("my_project")
except PackageNotFoundError:   # running from a source checkout, not installed
    __version__ = "0.0.0"

__all__ = [
    "getData",
    "vSWIM",
    "backmap",
]