from setuptools import setup, find_packages


with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="datasets2",
    version="0.1.1",
    description="Add-ons to the huggingface `datasets`",
    url="https://github.com/zzsi/datasets2",
    author="ZZ Si",
    author_email="zhangzhang.si@gmail.com",
    license="MIT",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
