---
title: Supercritical CO2 Energy Storage (sCO2ES) Reference
description: A 1-D transient solver for supercritical CO2 packed bed thermal energy storage implemented in Python.
hide:
- navigation
---

# Reference

This is the reference for the Supercritical CO₂ Energy Storage (sCO₂ES) package (`sco2es`). The solvers are written 
entirely in Python, using [Numba](https://numba.readthedocs.io/en/stable/) wherever possible for just-in-time (JIT) compilation and parallelization 
to maximize performance. 

By default, [`sco2es.PackedBed`][] uses [CoolProp](https://github.com/coolprop/coolprop) to calculate the properties of 
CO~2~ and assumes that the solid particles are made of alumina, for which empirical correlations are used to calculate 
relevant properties. To use a material other than alumina, create a class matching the protocol 
[`sco2es.SolidPropsInterface`][] and set the [`sco2es.PackedBed.solid`][] attribute. 

!!! Warning
    The model was intended for use with sCO~2~ as the heat transfer fluid. The `sco2es.PackedBed.fluid` attribute can be 
    set to any `CoolProp.AbstractState` object, but the code has not been tested with any other fluid.

---

::: sco2es
