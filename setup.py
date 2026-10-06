from setuptools import find_packages, setup

setup(
    name="ae_damage_id",  
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "vallenae==0.11.0",
        "seaborn==0.13.2",
        "scikit-learn==1.6.1",
        "scipy==1.11.3",
        "scikit-image==0.25.2",
        "tensorflow==2.19.0"
    ],
)
