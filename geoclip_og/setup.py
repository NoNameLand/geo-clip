# ...existing code...
from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf8") if (HERE / "README.md").exists() else ""

setup(
    name="geoclip_og",
    version="0.0.1",
    description="GeoCLIP - location encoder and geospatial CLIP utilities",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Your Name",
    license="MIT",
    packages=find_packages(exclude=("tests", "experiments", "docs")),
    include_package_data=True,
    install_requires=[
        # keep this minimal; add real deps as needed
        "torch>=1.10",
    ],
    python_requires=">=3.8",
)
# ...existing code...