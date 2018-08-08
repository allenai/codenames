"""setup.py file for packaging ``codenames``"""

from setuptools import setup, find_packages


with open('readme.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='codenames',
    version='0.0.1',
    description="Codenames hackathon 2018 project!",
    long_description=readme,
    url='http://github.com/allenai/codenames',
    author='Allen Institute for Artificial Intelligence',
    author_email='pradeepd@allenai.org',
    license='Apache',
    packages=find_packages(),
    python_requires='>=3.6',
    zip_safe=False
)
