from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gencome',
      version='0.1',
      description='GENCOME - GENetic COunt-based Measure',
      url='https://github.com/Ryookia/MgrRawData.git',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='',
      author_email='',
      license='',
      package_dir={"": "src"},
      packages=find_packages("src"),
      install_requires=[
          'numpy>=1.18.1',
          'pandas>=1.0.2',
          'deap>=1.3.1',
          'pygraphviz>=1.5',
          'scipy>=1.5.1',
      ],
      scripts=[
          'scripts/gencome_runner.py'
      ],
      zip_safe=False,
      python_requires='>=3.6',
)
