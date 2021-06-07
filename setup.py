from setuptools import setup, find_packages

setup(
    name='SMILES-RNN',
    version='0.1',
    packages=['model'],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A Generic SMILES-RNN, modified from and based off reinvent 2.0',
    scripts=['sample_model.py', 'randomize_smiles.py', 'train_prior.py', 'fine_tune.py', 'reinforcement_learning.py']
)
