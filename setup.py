import setuptools
from pathlib import Path


# Reading the long description from README.md
def read_long_description():
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return "DeepCode: Open Agentic Coding (Paper2Code & Text2Web & Text2Backend)"


# Retrieving metadata from __init__.py
def retrieve_metadata():
    vars2find = ["__author__", "__version__", "__url__", "__description__"]
    vars2readme = {}
    try:
        with open("./__init__.py", encoding="utf-8") as f:
            for line in f.readlines():
                for v in vars2find:
                    if line.startswith(v):
                        line = (
                            line.replace(" ", "")
                            .replace('"', "")
                            .replace("'", "")
                            .strip()
                        )
                        vars2readme[v] = line.split("=")[1]
    except FileNotFoundError:
        raise FileNotFoundError("Metadata file './__init__.py' not found.")

    # Checking if all required variables are found
    missing_vars = [v for v in vars2find if v not in vars2readme]
    if missing_vars:
        raise ValueError(
            f"Missing required metadata variables in __init__.py: {missing_vars}"
        )

    return vars2readme


# Reading dependencies from requirements.txt
def read_requirements():
    deps = []
    try:
        with open("./requirements.txt", encoding="utf-8") as f:
            deps = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]
    except FileNotFoundError:
        print(
            "Warning: 'requirements.txt' not found. No dependencies will be installed."
        )
    return deps


metadata = retrieve_metadata()
long_description = read_long_description()
requirements = read_requirements()

# All dependencies are required for full functionality
# No optional dependencies to ensure complete installation

setuptools.setup(
    name="deepcode",
    url=metadata["__url__"],
    version=metadata["__version__"],
    author=metadata["__author__"],
    description=metadata["__description__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(
        exclude=("tests*", "docs*", ".history*", ".git*", ".ruff_cache*")
    ),  # Automatically find packages
    py_modules=["deepcode"],  # Include the main deepcode.py module
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.9",
    install_requires=requirements,  # All dependencies are required
    include_package_data=True,  # Includes non-code files from MANIFEST.in
    entry_points={
        "console_scripts": [
            "deepcode=deepcode:main",  # Command line entry point
        ],
    },
    project_urls={  # Additional project metadata
        "Documentation": metadata.get("__url__", ""),
        "Source": metadata.get("__url__", ""),
        "Tracker": f"{metadata.get('__url__', '')}/issues"
        if metadata.get("__url__")
        else "",
    },
    package_data={
        "deepcode": [
            "config/*.yaml",
            "prompts/*",
            "schema/*",
            "*.png",
            "*.md",
        ],
    },
)
