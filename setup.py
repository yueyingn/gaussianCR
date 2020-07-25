from setuptools import setup
from setuptools import find_packages

setup(
    name='gaussianCR',
    version='0.0',
    description='Implementation of constrained realization to Gaussian promordial density fields',
    author='Yueying Ni',
    author_email='yueyingn@andrew.cmu.edu',
    packages=find_packages(),
    python_requires='>=3',
    install_requires=[
        'numpy',
        'scipy',
    ],
    scripts=[
    ]
)



