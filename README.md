# Supercritical Carbon Dioxide Energy Storage (sCO₂-ES)

This code is intended for modeling of energy storage for supercritical carbon dioxide (sCO₂) 
packed bed thermal energy storage systems. This work is based on the model detailed by Battisti et al.[^1].

## Performance

This code uses Numba for optimization of functions and utilizes NumPy's linear algebra routines
and SciPy's sparse linear algebra routines. For optimal performance, a version of SciPy built against a well
optimized LAPACK/BLAS library is required. Therefore, it is recommended to run this code using the 
Anaconda distribution of SciPy, which is built against Intel’s MKL[^2].

[^1]: F. G. Battisti, L. A. de Araujo Passos, and A. K. da Silva, "Performance mapping of packed-bed thermal energy 
storage systems for concentrating solar-powered plants using supercritical carbon dioxide," Applied Thermal Engineering, 
vol. 183, 2021, doi: 10.1016/j.applthermaleng.2020.116032.

[^2]: https://numba.readthedocs.io/en/stable/user/performance-tips.html
