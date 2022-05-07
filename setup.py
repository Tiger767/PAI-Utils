import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='paiutils',
    version='4.1.0',
    author='Travis Hammond',
    description='An artificial intelligence utilities package '
                'built to remove the delays of machine learning research.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Tiger767/PAI-Utils',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    install_requires=['numpy>=1', 'h5py>=2', 'matplotlib>=3',
                      'scikit-learn', 'opencv-python>=4'],
    extras_require={'tf': ['tensorflow>=2.1'],
                    'tf_gpu': ['tensorflow-gpu>=2.1'],
                    'tfp': ['tensorflow_probability>=0.11.1'],
                    'gym': ['gym>=0.17.1'],
                    'pa': ['pyaudio>=0.2'],
                    'wv': ['webrtcvad>=2.0']}
)
