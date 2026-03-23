from setuptools import setup, find_packages

setup(
    name="optlib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy"],
    author="Seu Nome",
    description="Biblioteca de algoritmos de otimização não linear (baseada em Ribeiro & Karas, 2012)",
    python_requires=">=3.8",
)
