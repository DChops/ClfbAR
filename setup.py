import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="clfbar",
    version="0.0.1",
    author="Dhruv Chopra",
    description="Classification By Association Rules Mining (CARS) Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    py_modules=["clfbar"],
    package_dir={'':'src/clfbar'},
    install_requires=[]
)