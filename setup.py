from setuptools import find_packages, setup

setup(
    name="abtests",
    packages=find_packages(include=["abtests"]),
    version="0.1.0",
    description="Python library for running A/B Tests",
    author="Buser Brasil Data Science Team",
    license="MIT",
    install_requires=["statsmodels"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
)
