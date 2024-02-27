from setuptools import setup, find_packages


__version__ = "1.0.0"
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="code_duality",
    version=__version__,
    author="Charles Murphy",
    author_email="charles.murphy.1@ulaval.ca",
    url="https://github.com/charlesmurphy1/midynet",
    license="MIT",
    description="Code for `Duality between predictability and reconstructability in complex systems`.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.9",
)
