# Supercritical Carbon Dioxide Energy Storage (sCO₂-ES)

This code is intended for modeling of energy storage for supercritical carbon dioxide (sCO₂). 
The following models are currently available:

- Packed bed thermal energy storage

## Performance

This code uses Numba for optimization of functions and utilizes NumPy's linear algebra routines
and SciPy's sparse linear algebra routines. For optimal performance, a version of SciPy built against a well
optimized LAPACK/BLAS library is required. Therefore, it is recommended to run this code using the 
Anaconda distribution of SciPy, which is built against Intel’s MKL[^1].

[^1]: https://numba.readthedocs.io/en/stable/user/performance-tips.html
