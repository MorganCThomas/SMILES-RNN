from setuptools import setup, find_packages

setup(
    name='SMILES-RNN',
    version='1.0',
    packages=['model'],
    license='MIT',
    author='Morgan Thomas',
    author_email='morganthomas263@gmail.com',
    description='A Generic SMILES-RNN, modified from and based off reinvent 2.0',
    scripts=['sample_model.py', 'train_prior.py', 'fine_tune.py', 'reinforcement_learning.py',
     'utility_scripts/deep_smiles.py', 'utility_scripts/randomize_smiles.py']
)
