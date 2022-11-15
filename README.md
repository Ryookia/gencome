# GENCOME - GENetic COunt-based Measures discovErer

**Note: the repository has been moved to: https://github.com/mochodek/gencome for further maintenance. 

GENCOME is a tool that allows discovering new count-based software measures based on the provided examples. Count-based measures have definitions consisting of two steps. In the first step, we apply rules to decide whether to count a given object or not, and in the second step we count the number of objects for which the answer was yes. A good example of count-based measure are lines of code (LOC). For instance, if you would like to count so-called non-commented lines of code (NCLOC) in a file you have to iterate over the lines in that file and for each of them evaluate the following rule "IF not commented AND not empty THEN true OTHERWISE false" and count the number of lines for which the answer was "true."

## Installation

Download or clone the repository, open terminal in the root directory of the project and run the following command (if you are using conda or other Python virtualization tools, please remember to activate your virtual environment before):

```bash
pip install -e .
```

## Usage

You need two csv files as an input for the tool. The first one, let's call it X, contains entries for objects (e.g., lines of code). The objects are grouped by the "id" column (e.g., the name of the source file they belong to) while the remaining columns are feautures (we accept numerical values) describing the object. The second file, let's call it Y, contains the output values for which the discovered measure should correlate to and again grouped by "id" (e.g., the effort to implement each of the source files). The tool will try to maximize the correlation between the X and Y by generating different measure definitions using Genetic Programming.

See the "examples" folder for a complete example on how to use the tool.
