from setuptools import setup, find_packages

setup(
    name='bart_survival',
    version='0.0.2',
    install_requires=[
        'arviz; python_version<="0.17.0',
        "numpy",
        "scipy",
        "typing",
        "pathlib",
        'pymc; python_version<="5.10.3"',
        'pymc_bart; python_version<="0.5.7"',
        'xarray; python_version<=2024.1.1',
        'pytensor<=2.18.6',
        "cloudpickle"
    ],
    packages=find_packages(
        # All keyword arguments below are optional:
        where='src',  # '.' by default
        include=['bart_survival*'],  # ['*'] by default
        # exclude=['mypackage.tests'],  # empty by default
    ),
    # include_package_data=True,

)