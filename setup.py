#!/usr/bin/env python

#import os
from setuptools import setup, find_packages
from subprocess import call

with open('README.md', 'r') as f:
      readme = f.read()

with open('requirements.txt') as f:
      required = f.read().splitlines()

setup(name='tf_course',
      version='1.0',
      description="Coursera's TensorFlow course materials",
      long_description=readme,
      author='Mateusz Sadowski-Zdunek',
      author_email='msadowskizdunek@gmail.com',
      packages=find_packages(),
      install_requires=required
     )

#Install requirements
#call(['pip', 'install', '-r', 'requirements.txt'])