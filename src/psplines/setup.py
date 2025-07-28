from setuptools import setup, find_packages

setup(
    name="psplines",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy>=1.21", "scipy>=1.7", "pandas>=1.3", "matplotlib>=3.4"],
    author="Your Name",
    author_email="your.email@example.com",
    description="P-spline smoothing for (x, y) data based on Practical Smoothing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/psplines",
)
