from setuptools import setup, find_packages

setup(
    name="hossam",
    version="0.0.1",
    description="Hossam Data Helper is a Python package that provides various functions to help you analyze data.",
    author="Lee Kwang-Ho",
    author_email="leekh4232@gmail.com",
    license="MIT",
    packages=find_packages(exclude=[]),
    keywords=["data", "analysis", "helper", "hossam", "이광호"],
    python_requires=">=3.11",
    zip_safe=False,
    url="https://github.com/leekh4232/hossam_data_helper",
    install_requires=[
        "tabulate",
        "pandas",
        "matplotlib",
        "seaborn",
        "statsmodels",
        "scipy",
        "pingouin",
        "scikit-learn",
        "imblearn",
        "pdoc3",
        "pmdarima",
        "prophet",
    ],
)
