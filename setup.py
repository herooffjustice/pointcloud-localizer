from setuptools import setup, find_packages

setup(
    name="pointcloud_localizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pytest",
    ],
    entry_points={
        "console_scripts": [
            "pointcloud-localizer=pointcloud_localizer.cli:main",
        ],
    },
    python_requires=">=3.8",
)