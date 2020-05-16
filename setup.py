import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="paiutils",
    version="2.1.1",
    author="Travis Hammond",
    description="An artificial intelligence utilities package built to remove the delays of machine learning research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tiger767/PAI-Utils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    python_requires='>=3.7',
    include_package_data=True,
    install_requires=['numpy>=1', 'h5py>=2', 'matplotlib>=3', 'tensorflow>=2', 'scikit-learn', 'opencv-python>=4']
)