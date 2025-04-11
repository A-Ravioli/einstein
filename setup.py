from setuptools import setup, find_packages

setup(
    name="einstein",
    version="0.1.0",
    description="AI Co-scientist system using AutoGen",
    author="Einstein Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "autogen-agentchat>=0.5.0",
        "autogen-ext>=0.5.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 