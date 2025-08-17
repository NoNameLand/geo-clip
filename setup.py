from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="geoclip",
    version="1.2.0",
    packages=find_packages(),
    description="GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VicenteVivan/geo-clip",
    author="Vicente Vivanco",
    author_email="vicente.vivancocepeda@ucf.edu",
    license="MIT",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "pytest-cov>=2.10.0",
        ]
    },
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision, geo-localization, clip, pytorch, deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/VicenteVivan/geo-clip/issues",
        "Source": "https://github.com/VicenteVivan/geo-clip",
        "Documentation": "https://github.com/VicenteVivan/geo-clip/tree/main/docs",
        "Paper": "https://arxiv.org/abs/2309.16020v2",
    },
)
 