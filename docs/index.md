---
hide:
- navigation
---

# Reference

This is the reference for the `packed_bed` module. The solvers are written entirely in Python, using 
[`numba`](https://numba.readthedocs.io/en/stable/) wherever possible for just-in-time (JIT) compilation and
parallelization to maximize performance.

By default, `PackedBedModel` uses [`CoolProp`](https://github.com/coolprop/coolprop) to calculate the properties of 
CO~2~ and assumes that the solid particles are made of alumina, for which empirical correlations are used to calculate 
relevant properties. To use custom routines for properties - for example, in the case of a different particle material -
`PackedBedModel` should be subclassed and [`calculate_fluid_props`](#packed_bed.PackedBedModel.calculate_fluid_props) 
and/or [`calculate_solid_props`](#packed_bed.PackedBedModel.calculate_solid_props) should be overridden.

---

::: packed_bed
