from setuptools import find_packages, setup

version = "0.1.0"

# Load requirements from requirements.txt
with open("requirements.txt", "r") as f:
    install_requires = [line.strip() for line in f if line.strip()]

setup(
    name="Ezspeech",
    version=version,
    description="A simple speech recognition and text-to-speech library",
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your_email@example.com",
    url="https://github.com/khanh14ph/Ezspeech",
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
