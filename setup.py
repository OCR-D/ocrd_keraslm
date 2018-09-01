# -*- coding: utf-8 -*-
"""
Installs:
    - keraslm-rate
"""
import codecs

from setuptools import setup, find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='ocrd_keraslm',
    version='0.1.0',
    description='keras language model',
    long_description=README,
    author='Konstantin Baierer, Kay-Michael Würzner',
    author_email='unixprog@gmail.com, wuerzner@gmail.com',
    url='https://github.com/OCR-D/ocrd_keraslm',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        'ocrd >= 0.8.0',
        'keras',
        'click',
        'numpy',
        'tensorflow',
    ],
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'keraslm-rate=ocrd_keraslm.scripts.run:cli',
            'ocrd-keraslm-rate=ocrd_keraslm.wrapper.cli:ocrd_keraslm_rate',
        ]
    },
)
