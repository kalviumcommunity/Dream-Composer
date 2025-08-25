from setuptools import setup, find_packages

setup(
    name="dream-composer",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pytest",
        "torch>=2.0.0",
    ],
)
