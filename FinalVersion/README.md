# kernelsmlProject


Kernels methods in Machine Learning project


### Dependancies


* Python 3.5 is recommended for Windows users (for CVXOPT, see infra).
* NumPy+MKL
* [CVXOPT](http://cvxopt.org/) in order to solve QPs. This package has known compatibility issues on Windows; Windows users should consider using Anaconda with a Python 3.5 environment and install this package and Numpy using `conda install cvxopt` instead of `pip`.
* [Numba](https://numba.pydata.org/) for computationally efficient numerical functions.
* [Joblib](https://pythonhosted.org/joblib/) for generic multithreading.


### Instructions


The submission files generate a `.csv` file in the current folder. This file is our submission file.


The main submission file may be ran from the current folder:
```
python start.py
```


In order to run the other submission scripts (in `submissionScripts`) and the test scripts (in `testScripts`), one needs to move the corresponding Python file to this folder (i.e. the folder containing `README.md`) so that it can load the package `kernelsmlProject`. For instance:
```
cp ./submissionScripts/submission_01_BoW.py ./submission_01_BoW.py
python submission_01_BoW.py
```
```
cp ./testScripts/tests_MultipleSpectrumGaussian.py ./tests_MultipleSpectrumGaussian.py
python tests_MultipleSpectrumGaussian.py
```


### Multithreading using joblib


This project uses Joblib as a multithreading library to speed generic computations up (e.g. when computing `K(x,y)` explicitely for each possible pair). Most of our kernels are more or less vectorized and do not make use of it. On Windows systems, no code should run outside a `if __name__ == '__main__':` loop, as explained [in the official documentation](https://pythonhosted.org/joblib/parallel.html). As a result, the multithreading features are disabled by default. One should precise `enable_joblib=True` when instanciating a `Kernel` object.
