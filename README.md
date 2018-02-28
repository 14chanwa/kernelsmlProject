# kernelsmlProject


Kernels methods in Machine Learning project


### Dependancies


* Python 3.5 is recommended for Windows users (for CVXOPT, see infra).
* NumPy+MKL
* [CVXOPT](http://cvxopt.org/) in order to solve QPs. This package has known compatibility issues on Windows; Windows users should consider using Anaconda with a Python 3.5 environment and install this package and Numpy using `conda install cvxopt` instead of `pip`.
* [Numba](https://numba.pydata.org/) for computationally efficient numerical functions.
* [Joblib](https://pythonhosted.org/joblib/) for generic multithreading.



### Multithreading using joblib


This project uses Joblib as a multithreading library to speed generic computations up (e.g. when computing `K(x,y)` explicitely for each possible pair). Most of our kernels are more or less vectorized and do not make use of it. On Windows systems, no code should run outside a `if __name__ == '__main__':` loop, as explained [in the official documentation](https://pythonhosted.org/joblib/parallel.html). As a result, the multithreading features are disabled by default. One should precise `enable_joblib=True` when instanciating a `Kernel` object.
