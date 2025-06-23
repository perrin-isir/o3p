# Install with 'pip install -e .'

from setuptools import setup, find_namespace_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="o3p",
    version=(Path(__file__).with_name("o3p") / "VERSION").read_text().strip(),
    description="o3p: a JAX-based library for offline and "
        "online off-policy reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/perrin-isir/o3p",
    packages=find_namespace_packages(),
    include_package_data=True,
    license="LICENSE",
)
