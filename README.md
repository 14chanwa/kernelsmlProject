# kernelsmlProject


Kernels methods in Machine Learning project


### Multithreading using joblib


This project uses 'joblib' as a multithreading library to speed computations up. On Windows systems, no code should run outside a `if __name__ == '__main__'` loop, as explained [in the official documentation](https://pythonhosted.org/joblib/parallel.html). As a result, the multithreading features are disabled by default. One should precise `enable_joblib=True` when instanciating a `Kernel` object.