from setuptools import setup, find_packages

setup(
    name="hossam",
    version="0.1.6",
    description="Hossam Data Helper is a Python package that provides various functions to help you analyze data.",
    author="Lee Kwang-Ho",
    author_email="leekh4232@gmail.com",
    license="MIT",
    packages=find_packages(exclude=[]),
    keywords=["data", "analysis", "helper", "hossam", "tensorflow", "이광호"],
    python_requires=">=3.10",
    zip_safe=False,
    url="https://github.com/leekh4232/hossam_data_helper",
    install_requires=[
        "tqdm",
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
        "graphviz",
        "dtreeviz",
        "pca",
        "statannotations",
        "pycallgraphix",
        "xgboost",
        "lightgbm",
        "tensorflow",
        "keras-tuner",
        "nltk",
        "contractions",
    ],
)
