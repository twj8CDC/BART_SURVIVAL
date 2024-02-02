from setuptools import setup, find_packages

setup(
    name='bart_survival',
    version='0.0.1.3',
    install_requires=[
        "arviz",
        "numpy",
        "scipy",
        "scikit-survival",
        "pymc_experimental",
        "typing",
        "pathlib",
        "pymc",
        "pymc_bart",
        "xarray",
        "pytensor",
        "cloudpickle"
    ],
    # packages=find_packages(
        # All keyword arguments below are optional:
        # where='src/',
        # include=[
        #     "bart_survival"
        # ],  
        # exclude=['tests'],  # empty by default
    # )
    
    # include_package_data=True,

)