from setuptools import setup

setup(
    packages=["smilesrnn"],
    scripts=[
        "scripts/sample_model.py",
        "scripts/train_prior.py",
        "scripts/fine_tune.py",
        "scripts/reinforcement_learning.py",
    ],
    install_requires=[
        "numpy",
        "torch",
        "deepsmiles",
        "tensorboard",
        "tqdm",
        "deepsmiles",
        "selfies",
        "google-auth",
        "promptsmiles",
    ]

)
