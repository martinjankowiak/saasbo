from setuptools import find_packages, setup

setup(
    name="saasbo",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["numpy>=1.20.0", "scipy>=1.7.0", "jax>=0.2.18", "numpyro>=0.7.2"],
)
