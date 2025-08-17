"""Deprecated shim for the original 'geoclip' package.

This repository now exposes the library under the name `geoclip_og` to avoid
conflicts with other packages. Importing `geoclip` will raise a clear error
directing users to the new package name.
"""
raise ImportError(
    "The 'geoclip' package in this repository is deprecated.\n"
    "Please import from 'geoclip_og' instead, for example: `from geoclip_og import GeoCLIP`")
