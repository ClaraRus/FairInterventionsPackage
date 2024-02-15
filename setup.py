from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Fair Interventions Python package.'
LONG_DESCRIPTION = 'Python package encapsulating pre-processing fairness interventions.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="FairInterventions",
    version=VERSION,
    author="Clara Rus",
    author_email="<c.a.rus@uva.nl>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['llvmlite', 'scikit-learn','numpy', 'pandas', 'rpy2==3.5.7'],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'

    keywords=['python'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)