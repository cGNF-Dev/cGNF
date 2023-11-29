from setuptools import setup, find_packages

setup(
    name='cGNF',
    version='0.6',  # start with a small number and increment it with every change
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
    author='Jesse Zhou',
    author_email='zhou.jesse2@gmail.com',
    description='The causal-Graphical Normalizing Flows (c-GNF) project primarily focuses on causal questions addressed through the use of normalizing flows',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/Jessezhou-1/cGNF',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

