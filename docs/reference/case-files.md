# Case Files

Simulation case files make running simulations more convenient by storing all the necessary information
in a single file. This includes the system geometry, initial and boundary conditions, and simulation parameters.
Simulations can be loaded from case files as shown below:

```py
import sco2es

sim = sco2es.PackedBed.load_case("case.yaml")
```

!!! Info "Current Limitations"
    Currently, case files can only be used to set up the simulation in a Python script. Future improvements include 
    storing charge and discharge cycle parameters in the case file to automate the entire simulation, as well as 
    running simulations from the command line.

## File format

Below is the reference YAML file format for the simulation case file:

```yaml title="reference_case.yaml"
system:
  packed-bed:
    length: 9.0  # [m]
    diameter: 3.0  # [m]
    particle-diameter: 0.003  # [m]
    void-fraction: 0.35
  wall:
    - thickness: 0.1  # [m]
      thermal-conductivity: 0.25  # [W/m/K]
      density: 250  # [kg/m^3]
      specific-heat: 1200  # [J/kg/K]

simulation:
  grid:
    axial-nodes: 1000
    wall-layer-nodes: 10
  initial-conditions:
    temperature: 298  # [K]
    pressure: 27.5e6  # [Pa]
  boundary-conditions:
    type: constant-temperature  # (1)!
    temperature: 298  # [K]
```

1. Constant temperature is currently the only supported boundary condition type
