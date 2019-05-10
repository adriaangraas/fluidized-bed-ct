# -*- coding: utf-8 -*-

__author__ = """Adriaan Graas"""
__email__ = 'adriaan.graas@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

from .bubblereactor import *