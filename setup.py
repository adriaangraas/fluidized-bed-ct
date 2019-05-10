#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os.path

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('CHANGELOG.md') as history_file:
#     history = history_file.read()

with open(os.path.join('fbrct', 'VERSION')) as version_file:
    version = version_file.read().strip()

requirements = [
    'scipy',
]

setup_requirements = []
test_requirements = []
dev_requirements = []

setup(
    author="Adriaan Graas",
    author_email='adriaan.graas@cwi.nl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    description="Fluidized Bed Reactor Computed Tomography",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    # long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='',
    name='fbrct',
    # packages=find_packages(include=[]),
    setup_requires=setup_requirements,
    # test_suite='tests',
    tests_require=test_requirements,
    extras_require={ 'dev': dev_requirements },
    url='https://github.com/adriaangraas/fluidized_bed_ct',
    version=version,
    zip_safe=False,
)
