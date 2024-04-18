from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.0.1'
DESCRIPTION = 'Higher level implementatin of the basics functions in openCv2 to accelerate the development of other projects.'

# Setting up
setup(
    name="gabriels_openCvLib",
    version=VERSION,
    author="Gabriel-Almeida-ECAT",
    author_email="<gabriel.oliveira.ga76@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['matplotlib','numpy','opencv','pandas','Pillow'],
    keywords=['python', 'image', 'OpenCv2'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
    ]
)
