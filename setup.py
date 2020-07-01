from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="causal-da",
    version="1.0.0",
    description='Implementation of Causal Mechanism Transfer.',
    long_description=readme,
    author='Takeshi Teshima',
    author_email='takeshi.78.teshima@gmail.com',
    url=
    'https://github.com/takeshi-teshima/few-shot-domain-adaptation-by-causal-mechanism-transfer',
    license=license,
    packages=find_packages(exclude=('docs', 'experiments')),
)
