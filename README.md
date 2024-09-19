# Supercritical Carbon Dioxide Energy Storage (sCO₂ES)

[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

`sco2es` is intended for modeling of energy storage for supercritical carbon dioxide (sCO₂) 
packed bed thermal energy storage systems. This work is based on the model detailed by Battisti et al.[^1].

> [!IMPORTANT]
> This software is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license 
> The software can be used for academic purposes, but if you are interested in using this
> software for commercial purposes, please contact the University of Central Florida Research Foundation's 
> [Technology Transfer Office](https://tt.research.ucf.edu/) to negotiate a commercial license.
> 
> Copyright © 2024 UCFRF, Inc. All Rights Reserved

## Performance

`sco2es` uses Numba for optimization of functions and utilizes NumPy's linear algebra routines
and SciPy's sparse linear algebra routines. For optimal performance, a version of SciPy built against a well
optimized LAPACK/BLAS library is required. Therefore, it is recommended to run this code using the 
Anaconda distribution of SciPy, which is built against Intel’s MKL[^2].

[^1]: F. G. Battisti, L. A. de Araujo Passos, and A. K. da Silva, "Performance mapping of packed-bed thermal energy 
storage systems for concentrating solar-powered plants using supercritical carbon dioxide," Applied Thermal Engineering, 
vol. 183, 2021, doi: 10.1016/j.applthermaleng.2020.116032.

[^2]: https://numba.readthedocs.io/en/stable/user/performance-tips.html
