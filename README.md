# kernelsmlProject


This repository is the handout for the project of the course [Kernels methods in Machine Learning](http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/course/2018mva/index.html) of the Master MVA, given by Jean-Philippe Vert and Julien Mairal. The objective of the project is to implement ``from scratch'' some algorithms and kernels in order to perform a classification task on strings (DNA sequences). The evaluation consists in a data challenge, in which the students of the course upload submissions of classification results on some test data.


Contributors: Quentin CHAN-WAI-NAM and Imke MEYER.


### Introduction


Kernels methods are useful tools in machine learning, as they enable to extend the reach of well-known linear classification and regression methods to more complicated data. The rough idea is to map the original data (which can be vectors, but also strings, graphs, etc...) to an other feature space, in which the classification/regression is well defined by a line. Kernels methods offer a rigorous mathematical framework to perform such operations, as well as mathematical results such as the kernel trick and the representer theorem, that enable us to build computationally efficient algorithms.



### Contents


We made a package `kernelsmlProject` that implement several linear classification algorithms (linear, logistic regressions, SVM) with compatibility with kernels methods. These algorithms can be found in `kernelsmlProject.algorithms`.


The core of the program consists in the implementation of kernels for vectors and strings. This part can be found in `kernelsmlProject.kernels`. We implemented several kernels for vectors (linear, polynomial, gaussian) and strings (spectrum and variants, substring...). The kernels can be centered in the feature space, i.e. the input data mapped to the feature space has mean zero.



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
