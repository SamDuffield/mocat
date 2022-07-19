
from setuptools import setup, find_namespace_packages

exec(open('mocat/version.py').read())

with open('README.md') as f:
    long_description = f.read()

setup(
    name='mocat',
    version=__version__,
    url='http://github.com/SamDuffield/mocat',
    author='Sam Duffield',
    python_requires='>=3.6',
    install_requires=['jax',
                      'jaxlib',
                      'matplotlib',
                      'decorator',
                      'numpy'],
    packages=find_namespace_packages(),
    author_email='sddd2@cam.ac.uk',
    description='All things Monte Carlo, written in JAX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT'
)


