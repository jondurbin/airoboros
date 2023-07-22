import os
from setuptools import setup

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")) as infile:
    long_description = infile.read()

setup(
    name="airoboros",
    version="2.0.5",
    description="Updated and improved implementation of the self-instruct system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jondurbin/airoboros",
    author="Jon Durbin",
    license="Apache 2.0",
    packages=["airoboros", "airoboros.instructors", "airoboros.instructors.prompts"],
    package_data={"airoboros": ["**/*.txt"]},
    include_package_data=True,
    install_requires=[
        "aiohttp[speedups]>=3.8",
        "backoff>=2.2",
        "requests>=2.28",
        "loguru>=0.7",
        "chromadb>=0.3.21",
        "langchain>=0.0.162",
        "sentence-transformers>=2.2.2",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "airoboros = airoboros.entrypoint:run"
        ]
    },
)
