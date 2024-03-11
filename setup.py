# -*- coding: utf-8 -*-
"""
Installs:
    - keraslm-rate
    - ocrd-keraslm-rate
"""
import codecs
import json

from setuptools import setup, find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()
with open('ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']

setup(
    name='ocrd_keraslm',
    version=version,
    description='character-level language modelling in Keras',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Robert Sachunsky, Konstantin Baierer, Kay-Michael Würzner',
    author_email='sachunsky@informatik.uni-leipzig.de, unixprog@gmail.com, wuerzner@gmail.com',
    url='https://github.com/OCR-D/ocrd_keraslm',
    license='Apache License 2.0',
    packages=find_packages(exclude=('test', 'repo', 'build')),
    install_requires=open('requirements.txt').read().split('\n'),
    extras_require={
        'plotting': [
            'sklearn',
            'matplotlib',
            'adjusttext',
            ]
    },
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'keraslm-rate=ocrd_keraslm.scripts.run:cli',
            'ocrd-keraslm-rate=ocrd_keraslm.wrapper.cli:ocrd_keraslm_rate',
        ]
    },
)
