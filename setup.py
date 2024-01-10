from setuptools import setup, find_packages

setup(
    name='cGNF',
    version='0.9.2',  # start with a small number and increment it with every change
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'networkx',
        'scikit-learn',
        'causalgraphicalmodels',
        'UMNN',
        'joblib',
    ],
    author='cGNF-Team',
    author_email='cgnf.team@gmail.com',
    description='A Python Module for Implementing causal-Graphical Normalizing Flows.',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/cGNF-Dev/cGNF',
    license='BSD License',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ],
)

