from setuptools import setup

setup(
    name="airoboros",
    version="0.0.2",
    description="Re-implementation of the self-instruct paper, with updated prompts, models, etc.",
    url="https://github.com/jondurbin/airoboros",
    author="Jon Durbin",
    license="Apache 2.0",
    packages=["airoboros"],
    install_requires=[
        "rouge-score>=0.1.2",
        "aiohttp>=3.8",
        "backoff>=2.2",
        "requests>=2.28",
        "loguru>=0.7",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
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
