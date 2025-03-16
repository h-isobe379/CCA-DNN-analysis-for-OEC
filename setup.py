from setuptools import setup, find_packages

setup(
    name='CCA-DNN-analysis-for-OEC',
    version='1.0.0',
    description='A framework for data processing, regularized CCA and DNN training.',
    author='Hiroshi Isobe',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy==1.26.4',
        'pandas==2.1.4',
        'matplotlib==3.8.0',
        'scipy==1.12.0',
        'scikit-learn==1.2.2',
        'chainer==7.8.1',
        'optuna==3.5.0',
        'tqdm==4.64.0',
    ],
)

