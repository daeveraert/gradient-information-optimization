import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grad-info-opt",
    version="0.1.2",
    author="Dante Everaert",
    author_email="dante.everaert@berkeley.edu",
    description="Implementation of Gradient Information Optimization for efficient and scalable training data selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/daeveraert/gradient-information-optimization",
    project_urls={
        "Bug Tracker": "https://github.com/daeveraert/gradient-information-optimization/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'jax>=0.3.25',
        'pyspark>=2.4.8',
        'numpy>=1.21.6',
        'sentence_transformers>=2.2.2',
        'jaxlib>=0.3.2',
        'pandas>=1.0.5']
)