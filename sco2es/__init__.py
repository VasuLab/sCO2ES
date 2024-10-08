__version__ = "0.1.0"

from .errors import StopCriterionError, ModelAssumptionError
from .properties import SolidProperties, Alumina

from typing import Annotated

import numpy as np
import numpy.typing as npt
from numba import njit, prange
import scipy
import CoolProp as CP
from ruamel.yaml import YAML


class PackedBed:
    r"""
    An implementation of the packed bed thermal energy storage model described by Battisti et al.[^1].

    [^1]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy storage
    systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal Engineering, vol.
    183, p. 116032, 2021.

    Assumptions:

    - Constant solid phase density
    - Constant wall thermal properties
    - Constant axial spacing
    - Constant diameter
    - Constant temperature exterior wall boundary condition

    !!! Warning
        The model was intended for use with sCO~2~ as the heat transfer fluid. The [`fluid`][sco2es.PackedBed.fluid] attribute can be
        set to any `CoolProp.AbstractState` object, but the code has not been tested with any other fluid.

    By default, `sco2es.PackedBed` uses [CoolProp](https://github.com/coolprop/coolprop) to calculate the properties of
    CO~2~ and assumes that the solid particles are made of [`Alumina`][sco2es.properties.Alumina], for which empirical correlations are used to calculate
    relevant properties. To use a material other than alumina, create a class matching the protocol
    [`SolidProperties`][sco2es.properties.SolidProperties] and set the [`solid`][sco2es.PackedBed.solid] attribute.

    !!! Info
        Arrays are annotated with dimensions as follows:

        - `N`: Number of time steps.
        - `Z`: Number of axial nodes.
        - `W`: Number of wall nodes.

    Attributes:
        axial_nodes: Number of nodes in the axial direction.
        wall_nodes: Number of nodes in the axial or radial directions for the walls and lids, respectively.

        z: Axial positions of the packed bed nodes from the charging inlet in the direction of flow [m].
        z_top_lid: Axial positions of top lid nodes from charging inlet in direction of flow [m].
        z_bottom_lid: Axial positions of bottom lid nodes from charging inlet in direction of flow [m].
        r_wall: Radial positions of wall nodes from centerline [m].

        T_f: Fluid temperature for bed nodes [K].
        T_s: Solid temperature for bed nodes [K].
        T_wall: Temperature of the wall nodes [K].
        T_top_lid: Temperature of the top lid nodes [K].
        T_bottom_lid: Temperature of the bottom lide nodes [K].

        i_f: Fluid enthalpy at the node centers [J/kg].
        cp_f: Fluid specific heat capacity at the node centers [J/kg⋅K].
        k_f: Fluid thermal conductivity at the node centers [W/m⋅K].
        rho_f: Fluid density at the node centers [kg/m^3^].

        A_cs: Cross-sectional area of the packed bed [m^2^].
        V_node: Volume of a packed bed node [m^3^].
        r_bound: Radial positions of wall cell boundaries from centerline [m].
        V_wall: Volume of the wall cells [m^3^].
        A_wall_z: Surface area of the wall cell boundary in the axial direction [m^2^].
        A_wall_r: Surface area of the wall cell boundary in the radial direction [m^2^].
    """

    fluid = CP.AbstractState("BICUBIC&HEOS", "CO2")
    """[`CoolProp.AbstractState`](http://www.coolprop.org/apidoc/CoolProp.AbstractState.html) object for accessing 
    fluid properties. By default, CO~2~ properties are tabulated for the Helmholtz-based equation of state (HEOS)
    and bicubic interpolation is used for performance."""

    solid: SolidProperties = Alumina
    """[`SolidPropsInterface`][sco2es.SolidPropsInterface] for calculating temperature-dependent properties of the 
    solid phase."""

    max_iter: int = 100
    """Maximum number of iterations for the loop."""

    atol_T_f: float = 0.05
    """Absolute tolerance for fluid temperature [ºC]."""
    atol_P: float = 0.1
    """Absolute tolerance for pressure [Pa]."""
    atol_T_s: float = 0.05
    """Absolute tolerance for solid temperature [ºC]."""

    rtol_T_wall: float = 0.05e-2
    """Relative tolerance for wall and lid temperatures."""
    rtol_i_f: float = 0.01e-2
    """Relative tolerance for fluid enthalpy."""
    rtol_rho_f: float = 0.1e-2
    """Relative tolerance for fluid density."""
    rtol_m_dot: float = 0.1e-2
    """Relative tolerance for mass flow rate."""
    rtol_h: float = 0.1e-2
    """Relative tolerance for volumetric and wall heat transfer coefficients."""

    def __init__(
            self,
            T_initial: float,
            P: float,
            L: float,
            D: float,
            d: float,
            eps: float,
            T_env: float,
            t_wall: npt.ArrayLike,
            k_wall: npt.ArrayLike,
            rho_wall: npt.ArrayLike,
            cp_wall: npt.ArrayLike,
            *,
            axial_nodes: int = 100,
            wall_layer_nodes: int | npt.ArrayLike = 10,
            solid: SolidProperties | None = None
    ):
        """
        Parameters:
            T_initial: Discharge temperature [K].
            P: Initial pressure [Pa].
            L: Domain length [m].
            D: Internal tank diameter [m].
            d: Particle diameter [m].
            eps: Void fraction.
            T_env: Temperature of the environment [K].
            t_wall: Thickness of each wall layer [m].
            k_wall: Thermal conductivity of each wall layer [W/m⋅K].
            rho_wall: Density of each wall layer [kg/m^3^].
            cp_wall: Specific heat capacity of each wall layer [J/kg⋅K].
            axial_nodes: Number of axial nodes.
            wall_layer_nodes: Number of nodes for each wall layer.
            solid: Solid properties interface.
        """

        # Simulation parameters
        self.axial_nodes: int = axial_nodes
        self.L: float = L
        self.dz: float = L / axial_nodes
        self.t: Annotated[npt.NDArray[float], "N"] = np.array([0])
        self.charging: Annotated[npt.NDArray[float], "N"] = np.array([], dtype=bool)
        self.z: Annotated[npt.NDArray[float], "Z"] = np.linspace(self.dz / 2, L - self.dz / 2, axial_nodes)

        if solid is not None:
            self.solid = solid

        # Packed bed parameters
        self.D: float = D
        self.d: float = d
        self.eps: float = eps
        self.A_cs: float = np.pi * self.D ** 2 / 4
        self.A_node_wall_intf: float = np.pi * self.D * self.dz
        self.V_node: float = np.pi * self.D ** 2 / 4 * self.dz

        # Fluid properties
        self.P_intf: Annotated[npt.NDArray[float], "N,Z+1"] = np.full((1, axial_nodes + 1), P)
        self.T_f: Annotated[npt.NDArray[float], "N,Z"] = np.full((1, axial_nodes), T_initial, dtype=float)
        self.i_f: Annotated[npt.NDArray[float], "Z"] = self.calculate_fluid_enthalpy(self.T_f[0], self._interp(self.P_intf[0]))
        self.cp_f: Annotated[npt.NDArray[float], "N,Z"] = np.zeros_like(self.T_f)
        self.k_f: Annotated[npt.NDArray[float], "Z"]
        self.rho_f: Annotated[npt.NDArray[float], "Z"]
        self.mu_f: Annotated[npt.NDArray[float], "Z"]
        self.k_f, self.rho_f, self.mu_f, self.cp_f[0] = self.calculate_fluid_props(
            self.i_f, self._interp(self.P_intf[0]))[1:]

        # Solid properties
        self.T_s: Annotated[npt.NDArray[float], "N,Z"] = np.full((1, axial_nodes), T_initial, dtype=float)
        self.rho_s: float = self.solid.density
        self.E_s: Annotated[npt.NDArray[float], "Z"] = self.solid.emissivity(self.T_s[0])
        self.k_s: Annotated[npt.NDArray[float], "Z"] = self.solid.thermal_conductivity(self.T_s[0])

        # Field variables
        self.m_dot: Annotated[npt.NDArray[float], "N,Z+1"] = np.zeros((1, axial_nodes + 1))  # Initially stationary
        self.k_eff: Annotated[npt.NDArray[float], "Z"]
        self.h_wall: Annotated[npt.NDArray[float], "Z"]
        self.h_v: Annotated[npt.NDArray[float], "Z"]
        self.k_eff, self.h_wall, self.h_v = self.calculate_heat_transfer_coeffs(
            self._interp(self.m_dot[0]), self.T_f[0], self.k_f, self.cp_f[0], self.mu_f, self.k_s, self.E_s)

        # Wall/lid parameters
        self.T_env: float = T_env

        t_wall = np.asarray(t_wall, dtype=float)
        k_wall = np.asarray(k_wall, dtype=float)
        rho_wall = np.asarray(rho_wall, dtype=float)
        cp_wall = np.asarray(cp_wall, dtype=float)

        self.D_outer: float = D + 2 * np.sum(t_wall)

        assert t_wall.size == k_wall.size == rho_wall.size == cp_wall.size
        if isinstance(wall_layer_nodes, int):
            wall_layer_nodes = np.full(t_wall.shape, wall_layer_nodes)

        self.wall_nodes: int = sum(wall_layer_nodes)
        self.dr: Annotated[npt.NDArray[float], "R"] = np.repeat(t_wall / wall_layer_nodes, wall_layer_nodes)

        # Determine wall grid properties
        dx_bound = np.add.accumulate(np.insert(self.dr, 0, 0))
        dx_center = self._interp(dx_bound)

        self.r_bound: Annotated[npt.NDArray[float], "W+1"] = self.D / 2 + dx_bound
        self.r_wall: Annotated[npt.NDArray[float], "W"] = self.D / 2 + dx_center
        self.z_top_lid: Annotated[npt.NDArray[float], "W"] = -dx_center[::-1]
        self.z_bottom_lid: Annotated[npt.NDArray[float], "W"] = L + dx_center

        self.A_wall_z: Annotated[npt.NDArray[float], "W"] = np.pi * (self.r_bound[1:]**2 - self.r_bound[:-1]**2)
        self.V_wall: Annotated[npt.NDArray[float], "W"] = self.A_wall_z * self.dz
        self.A_wall_r: Annotated[npt.NDArray[float], "W+1"] = 2 * np.pi * self.dz * self.r_bound

        self.V_bottom_lid: Annotated[npt.NDArray[float], "W"] = self.A_cs * np.diff(dx_bound)
        self.V_top_lid: Annotated[npt.NDArray[float], "W"] = self.V_bottom_lid[::-1]

        # Initialize wall/lid temperature distribution
        self.T_wall: Annotated[npt.NDArray[float], "N,Z,W"] = np.empty((1, self.axial_nodes, self.wall_nodes), dtype=float)
        self.T_top_lid: Annotated[npt.NDArray[float], "N,W"] = np.empty((1, self.wall_nodes), dtype=float)
        self.T_bottom_lid: Annotated[npt.NDArray[float], "N,W"] = np.empty((1, self.wall_nodes), dtype=float)

        # Calculate heat flux through lid and walls
        R_lid = t_wall / k_wall  # Convection/conduction resistance
        Q_lid = (T_initial - T_env) / np.sum(R_lid)  # Heat flux through the lid
        T_lid_bound = np.array([T_initial, *(T_initial - np.add.accumulate(Q_lid * R_lid))])

        r_layer_bound = D/2 + np.array([0, *np.add.accumulate(t_wall)])
        R_wall = np.log(r_layer_bound[1:] / r_layer_bound[:-1]) / (2 * np.pi * k_wall * L)
        Q_wall = (T_initial - T_env) / np.sum(R_wall)
        T_wall_bound = [T_initial, *(T_initial - np.add.accumulate(Q_wall * R_wall))]

        # Calculate coordinates and temperature profiles for walls and lids
        T_wall = []
        T_lid = []

        for i in range(wall_layer_nodes.size):
            dx = t_wall[i] / (2 * wall_layer_nodes[i]) * (1 + 2 * np.arange(wall_layer_nodes[i]))
            T_wall.append(T_wall_bound[i] -
                          Q_wall * np.log(1 + dx / r_layer_bound[i]) / (2 * np.pi * L * k_wall[i]))
            T_lid.append(T_lid_bound[i] + np.diff(T_lid_bound)[i] * dx / t_wall[i])

        T_lid = np.array(T_lid).flatten()
        T_wall = np.array(T_wall).flatten()
        self.T_top_lid[0] = T_lid[::-1]
        self.T_bottom_lid[0] = T_lid
        self.T_wall[0] = T_wall

        # Fill wall/lid properties
        self.k_wall: Annotated[npt.NDArray[float], "W"] = np.repeat(k_wall, wall_layer_nodes)
        self.rho_wall: Annotated[npt.NDArray[float], "W"] = np.repeat(rho_wall, wall_layer_nodes)
        self.cp_wall: Annotated[npt.NDArray[float], "W"] = np.repeat(cp_wall, wall_layer_nodes)

        self.k_bottom_lid: Annotated[npt.NDArray[float], "W"] = self.k_wall
        self.rho_bottom_lid: Annotated[npt.NDArray[float], "W"] = self.rho_wall
        self.cp_bottom_lid: Annotated[npt.NDArray[float], "W"] = self.cp_wall

        self.k_top_lid: Annotated[npt.NDArray[float], "W"] = self.k_wall[::-1]
        self.rho_top_lid: Annotated[npt.NDArray[float], "W"] = self.rho_wall[::-1]
        self.cp_top_lid: Annotated[npt.NDArray[float], "W"] = self.cp_wall[::-1]

        # Energy loss
        self.E_loss_top_lid: Annotated[npt.NDArray[float], "N-1"] = np.empty((1, 0), dtype=float)
        self.E_loss_bottom_lid: Annotated[npt.NDArray[float], "N-1"] = np.empty((1, 0), dtype=float)
        self.E_loss_walls: Annotated[npt.NDArray[float], "N-1"] = np.empty((1, 0), dtype=float)

    @classmethod
    def load_case(cls, case_file: str):
        """
        Loads a packed bed simulation case file in YAML format.

        Parameters:
            case_file: Filepath to the simulation case file.
        """

        yaml = YAML(typ="safe")
        with open(case_file, "r") as f:
            case = yaml.load(f)

        if case["simulation"]["boundary-conditions"]["type"] != "constant-temperature":
            raise NotImplementedError("Only constant-temperature boundary conditions are currently supported.")

        wall = case["system"]["wall"]
        if not wall:  # Check for empty list
            raise ValueError("At least one wall layer must be defined in the case file.")

        return cls(
            T_initial=case["simulation"]["initial-conditions"]["temperature"],
            P=case["simulation"]["initial-conditions"]["pressure"],
            L=case["system"]["packed-bed"]["length"],
            D=case["system"]["packed-bed"]["diameter"],
            d=case["system"]["packed-bed"]["particle-diameter"],
            eps=case["system"]["packed-bed"]["void-fraction"],
            T_env=case["simulation"]["boundary-conditions"]["temperature"],
            t_wall=[layer["thickness"] for layer in wall],
            k_wall=[layer["thermal-conductivity"] for layer in wall],
            rho_wall=[layer["density"] for layer in wall],
            cp_wall=[layer["specific-heat"] for layer in wall],
            axial_nodes=case["simulation"]["grid"]["axial-nodes"],
            wall_layer_nodes=case["simulation"]["grid"]["wall-layer-nodes"]
        )

    @staticmethod
    def _interp(x):
        """
        Returns the interpolated quantity at fluid node centers from the quantity at the interfaces.
        """
        return (x[:-1] + x[1:]) / 2

    def time_index(self, s: float = 0, *, m: float = 0, h: float = 0):
        """
        A function for retrieving the index of the simulation time step with the closest elapsed time
        to the given time.
        """
        return np.argmin(np.abs(s + 60 * (m + 60 * h) - self.t))

    @property
    def E_loss_total(self):
        """Total energy loss at each timestep."""
        return self.E_loss_top_lid + self.E_loss_bottom_lid + self.E_loss_walls

    def advance(
            self,
            T_inlet: float,
            P_inlet: float,
            m_dot_inlet: float,
            t_max: float = 12*60*60,
            *,
            T_outlet_stop: float = None,
            dt: float = 10,
            discharge: bool = False,
    ):
        """
        Advances the simulation until the charge/discharge stopping criterion is satisfied.

        Parameters:
            T_inlet: Inlet fluid temperature [K].
            P_inlet: Pressure at the inlet [Pa].
            m_dot_inlet: Mass flow rate of the inlet stream [kg/s].
            t_max: Maximum time advancement [s].
            T_outlet_stop: Outlet fluid temperature [K] condition for stopping.
            dt: Time step [s].
            discharge: Flag indicating that the system is discharging.

        Returns:
            Time at which the charging/discharging stop criterion was satisfied.

        Raises:
            ModelAssumptionError: Raised if the Biot number exceeds the acceptable threshold ($Bi > 0.1$).
            StopCriterionError: Raised if the stop criterion are not met within the maximum allowed
                charging/discharging time (`t_max`).
        """

        t_start = self.t[-1]
        while (t_elapsed := self.t[-1] - t_start) < t_max:
            self.step(T_inlet, P_inlet, m_dot_inlet, dt, discharge=discharge)

            if T_outlet_stop is not None:
                # Charge stopping criterion
                if not discharge:
                    if self.T_f[-1, -1] >= T_outlet_stop:
                        return t_elapsed

                # Discharge stopping criterion
                else:
                    if self.T_f[-1, 0] <= T_outlet_stop:
                        return t_elapsed
        else:
            if T_outlet_stop is not None:
                raise StopCriterionError(
                    f"{'Charging' if not discharge else 'Discharging'} stop criterion not satisfied within "
                    f"the maximum allowed time."
                )

            return t_elapsed

    def step(self, T_inlet: float, P_inlet: float, m_dot_inlet: float, dt, *, discharge: bool = False):
        """
        Calculates the state of the packed bed at the next time step using an iterative algorithm.

        Parameters:
            T_inlet: Temperature of the inlet stream [K].
            P_inlet: Pressure at the inlet [Pa].
            m_dot_inlet: Mass flow rate of the inlet stream [kg/s].
            dt: Time step [s].
            discharge: Flag indicating that the system is discharging.

        Returns:
            Number of iterations required for convergence.
        """

        Bi = self.biot_number(self.h_v, self.d, self.eps, self.k_s)
        if not np.all(Bi <= 0.1):  # Check lumped capacitance assumption
            raise ModelAssumptionError("Biot number exceeded acceptable threshold (Bi > 0.1).")

        i_inlet = self.calculate_fluid_enthalpy([T_inlet], [P_inlet])[0]

        # Next iteration state arrays
        P_intf = np.copy(self.P_intf[-1])  # Pressure at node interfaces
        P_intf[0 if not discharge else -1] = P_inlet  # Set inlet pressure
        m_dot = np.copy(self.m_dot[-1])  # Mass flow rate at node interfaces
        m_dot[0 if not discharge else -1] = m_dot_inlet  # Set inlet mass flow rate
        i_f = np.copy(self.i_f)
        T_f = np.copy(self.T_f[-1])
        rho_f = np.copy(self.rho_f)
        T_s = np.copy(self.T_s[-1])
        T_wall = np.copy(self.T_wall[-1])
        T_top_lid = np.copy(self.T_top_lid[-1])
        T_bottom_lid = np.copy(self.T_bottom_lid[-1])
        k_eff = np.copy(self.k_eff)
        h_v = np.copy(self.h_v)
        h_wall = np.copy(self.h_wall)

        for b in range(self.max_iter):
            # Previous iteration state arrays
            P_intf_prev = np.copy(P_intf)
            m_dot_prev = np.copy(m_dot)
            i_f_prev = np.copy(i_f)
            T_f_prev = np.copy(T_f)
            rho_f_prev = np.copy(rho_f)
            T_s_prev = np.copy(T_s)
            T_wall_prev = np.copy(T_wall)
            T_top_lid_prev = np.copy(T_top_lid)
            T_bottom_lid_prev = np.copy(T_bottom_lid)
            h_wall_prev = np.copy(h_wall)
            h_v_prev = np.copy(h_v)

            # Solve for fluid enthalpy and solid temperature
            g = T_f_prev / i_f_prev  # Calculate temperature-enthalpy coupling factor
            alpha1, alpha2 = self.solid.internal_energy_linear_coeffs(T_s_prev)
            e_s_0 = self.solid.internal_energy(self.T_s[-1])

            if not discharge:
                i_f, T_s = self.solve_fluid_solid_bed(
                    self.i_f, i_inlet, g,
                    e_s_0, alpha1, alpha2,
                    rho_f, self.rho_f, self.rho_s,
                    P_intf, self.P_intf[-1],
                    m_dot,
                    T_wall[:, 0], self.T_top_lid[-1, -1], self.T_bottom_lid[-1, 0],
                    k_eff, h_wall, h_v,
                    self.A_node_wall_intf, self.A_cs, self.V_node, self.eps, self.dz,
                    dt
                )
            else:
                i_f, T_s = self.solve_fluid_solid_bed(
                    self.i_f[::-1], i_inlet, g[::-1],
                    e_s_0[::-1], alpha1[::-1], alpha2[::-1],
                    rho_f[::-1], self.rho_f[::-1], self.rho_s,
                    P_intf[::-1], self.P_intf[-1, ::-1],
                    m_dot[::-1],
                    T_wall[::-1, 0], self.T_bottom_lid[-1, 0], self.T_top_lid[-1, -1],
                    k_eff[::-1], h_wall[::-1], h_v[::-1],
                    self.A_node_wall_intf, self.A_cs, self.V_node, self.eps, self.dz,
                    dt
                )
                i_f = i_f[::-1]
                T_s = T_s[::-1]

            # Update fluid and solid thermodynamic properties
            T_f, k_f, rho_f, mu_f, cp_f = self.calculate_fluid_props(i_f, self._interp(P_intf))
            E_s = self.solid.emissivity(T_s)
            k_s = self.solid.thermal_conductivity(T_s)

            # Update mass flow rate and pressure at interfaces
            if not discharge:
                m_dot[1:] = m_dot[0] - np.add.accumulate(
                    self.eps * self.V_node * (rho_f - self.rho_f) / dt
                )
            else:
                m_dot[-2::-1] = m_dot[-1] - np.add.accumulate(
                    self.eps * self.V_node * (rho_f[::-1] - self.rho_f[::-1]) / dt
                )

            G = self._interp(m_dot) / (self.eps * self.A_cs)  # Effective mass flow rate per cross-section

            if not discharge:
                P_intf[1:] = P_intf[0] - np.add.accumulate(
                    self.pressure_drop(self.dz, rho_f, mu_f, G, self.eps, self.d)
                )
            else:
                P_intf[-2::-1] = P_intf[-1] - np.add.accumulate(
                    self.pressure_drop(self.dz, rho_f[::-1], mu_f[::-1], G[::-1], self.eps, self.d)
                )

            # Update heat transfer coefficients
            k_eff, h_wall, h_v = self.calculate_heat_transfer_coeffs(
                self._interp(m_dot), T_f, k_f, cp_f, mu_f, k_s, E_s
            )

            # Solve for lid/wall temperatures
            T_top_lid = self.solve_lid_temperature(
                self.T_top_lid[-1], T_f[0], self.T_env, h_wall[0],
                self.k_top_lid, self.rho_top_lid, self.cp_top_lid,
                self.z_top_lid, self.V_top_lid, self.A_cs, dt,
                reverse=False
            )

            T_bottom_lid = self.solve_lid_temperature(
                self.T_bottom_lid[-1], T_f[-1], self.T_env, h_wall[-1],
                self.k_bottom_lid, self.rho_bottom_lid, self.cp_bottom_lid,
                self.z_bottom_lid, self.V_bottom_lid, self.A_cs, dt,
                reverse=True
            )

            T_wall = self.solve_wall_temperature(
                self.T_wall[-1], T_f, self.T_env, h_wall,
                self.k_wall, self.rho_wall, self.cp_wall,
                self.r_wall, self.dr, self.dz,
                self.V_wall, self.A_wall_r, self.A_wall_z, dt
            )

            E_loss_top_lid = (self.k_top_lid[0] * np.pi * self.D**2 / 4
                              * (T_top_lid[0] - self.T_env) / (self.dr[-1] / 2))
            E_loss_bottom_lid = (self.k_bottom_lid[-1] * np.pi * self.D**2 / 4
                                 * (T_bottom_lid[-1] - self.T_env) / (self.dr[-1] / 2))
            E_loss_walls = self.k_wall[-1] * self.D_outer * self.dz * np.sum(
                (T_wall[:, -1] - self.T_env) / (self.dr[-1] / 2)  # Shouldn't heat loss through the wall use ln(r2/r1)?
            )

            # Check convergence
            converged = np.all([
                np.all(np.abs(P_intf - P_intf_prev) <= self.atol_P),
                np.all(np.abs(T_f - T_f_prev) <= self.atol_T_f),
                np.all(np.abs(T_s - T_s_prev) <= self.atol_T_s),
                np.all(np.abs(T_wall - T_wall_prev) <= self.rtol_T_wall * T_wall),
                np.all(np.abs(T_top_lid - T_top_lid_prev) <= self.rtol_T_wall * T_top_lid),
                np.all(np.abs(T_bottom_lid - T_bottom_lid_prev) <= self.rtol_T_wall * T_bottom_lid),
                np.all(np.abs(i_f - i_f_prev) <= self.rtol_i_f * i_f),
                np.all(np.abs(rho_f - rho_f_prev) <= self.rtol_rho_f * rho_f),
                np.all(np.abs(m_dot - m_dot_prev) <= self.rtol_m_dot * m_dot),
                np.all(np.abs(m_dot - m_dot_prev) <= self.rtol_m_dot * m_dot),
                np.all(np.abs(h_v - h_v_prev) <= self.rtol_h * h_v),
                np.all(np.abs(h_wall - h_wall_prev) <= self.rtol_h * h_wall),
            ])

            if converged:
                self.t = np.append(self.t, self.t[-1] + dt)
                self.charging = np.append(self.charging, not discharge)
                self.P_intf = np.append(self.P_intf, [P_intf], axis=0)
                self.T_f = np.append(self.T_f, [T_f], axis=0)
                self.T_s = np.append(self.T_s, [T_s], axis=0)
                self.T_wall = np.append(self.T_wall, [T_wall], axis=0)
                self.T_top_lid = np.append(self.T_top_lid, [T_top_lid], axis=0)
                self.T_bottom_lid = np.append(self.T_bottom_lid, [T_bottom_lid], axis=0)
                self.E_loss_top_lid = np.append(self.E_loss_top_lid, [E_loss_top_lid])
                self.E_loss_bottom_lid = np.append(self.E_loss_bottom_lid, [E_loss_bottom_lid])
                self.E_loss_walls = np.append(self.E_loss_walls, [E_loss_walls])
                self.cp_f = np.append(self.cp_f, [cp_f], axis=0)
                self.m_dot = np.append(self.m_dot, [m_dot], axis=0)
                self.i_f = i_f
                self.rho_f = rho_f
                self.h_v = h_v
                self.h_wall = h_wall
                self.k_eff = k_eff

                return b + 1

        raise Exception("Maximum number of iterations reached without convergence.")

    @staticmethod
    @njit(parallel=True)
    def solve_fluid_solid_bed(
            i_f_0, i_inlet, g,  # Fluid enthalpy
            e_s_0, alpha1, alpha2,  # Solid internal energy
            rho_f, rho_0, rho_s,  # Density
            P_intf, P_intf_0,  # Pressure
            m_dot,  # Mass flow
            T_wall, T_lid_inlet, T_lid_outlet,  # Temperature BCs
            k_eff, h_wall, h_v,  # Heat transfer terms
            A_wall, A_cs, V, eps, dz,  # Geometry
            dt
    ):
        """
        Solves for the fluid enthalpy and solid temperature for the next time step.

        Parameters:
            i_f_0: Fluid enthalpy for the previous time step [J/kg].
            i_inlet: Fluid enthalpy at the inlet [J/kg].
            g: Fluid temperature-enthalpy coupling factor [K⋅kg/J].
            e_s_0: Solid internal energy for the previous time step [J/kg].
            alpha1: First linearized parameter for solid internal energy.
            alpha2: Second linearized parameter for solid internal energy.
            rho_f: Fluid density estimates for the next time step [kg/m^3^].
            rho_0: Fluid densities for the previous time step [kg/m^3^].
            rho_s: Density of the solid [kg/m^3^].
            P_intf: Pressure estimate at the interfaces for the next time step [Pa].
            P_intf_0: Pressure at the interfaces for the previous time step [Pa].
            m_dot: Mass flow rate estimate at node interfaces [kg/s].
            T_wall: Wall interface temperature [K].
            T_lid_inlet: Temperature of the lid at the inlet [K].
            T_lid_outlet: Temperature of the lid at the outlet [K].
            k_eff: Effective thermal conductivity [W/m⋅K]
            h_wall: Wall heat transfer coefficient [W/m^2^⋅K].
            h_v: Volumetric heat transfer coefficient [W/m^3^⋅K].
            A_wall: Surface area of the node wall boundary [m^2^].
            A_cs: Cross-sectional area [m^2^].
            V: Node volume [m^3^].
            eps: Void fraction [-].
            dz: Axial node spacing [m].
            dt: Time step [s].

        Returns:
            i_f: Fluid enthalpy for the next time step [J/kg].
            T_s: Solid temperature for the next time step [K].
        """
        # Matrix setup
        n = i_f_0.size
        a = np.zeros((2*n, 2*n))
        b = np.zeros(2*n)

        # Calculate effective thermal conductivity at interfaces
        k_eff_intf = 2 * k_eff[:-1] * k_eff[1:] / (k_eff[:-1] + k_eff[1:])

        # Calculate density at interfaces
        rho_intf = np.empty(n+1)
        rho_intf[1:-1] = (rho_f[:-1] + rho_f[1:]) / 2  # Interpolated density
        rho_intf[0] = rho_f[0] + 0.5 * (rho_f[0] - rho_f[1])  # Extrapolated density to dz/2 from center
        rho_intf[-1] = rho_f[-1] + 0.5 * (rho_f[-1] - rho_f[-2])

        # Calculate pressure at node centers
        P = (P_intf[:-1] + P_intf[1:]) / 2
        P_0 = (P_intf_0[:-1] + P_intf_0[1:]) / 2

        # Fill matrices
        for i in prange(n):
            f = i
            s = i + n

            # Fluid enthalpy equation
            a[f, f] = (
                eps * V * rho_f[i] / dt
                + m_dot[i+1]
                + h_v[i] * V * g[i]  # Heat transfer to solid phase
                + g[i] * h_wall[i] * A_wall  # Heat loss to walls
            )
            a[f, s] = -h_v[i] * V  # Heat transfer to solid phase

            b[f] = (
                eps * V * rho_0[i] * i_f_0[i] / dt
                + h_wall[i] * A_wall * T_wall[i]  # Heat loss to walls
                + eps * V * (P[i] - P_0[i]) / dt
                + m_dot[i+1] / rho_intf[i+1] * (P_intf[i+1] - P[i])
                + m_dot[i] / rho_intf[i] * (P[i] - P_intf[i])
            )

            if i != 0:  # Energy flow from upstream node
                a[f, f-1] = -m_dot[i]
            else:
                b[f] += m_dot[i] * i_inlet

            if i == 0:  # Top lid boundary
                a[f, f] += g[0] * h_wall[0] * A_cs
                b[f] += h_wall[0] * A_cs * T_lid_inlet

            if i == n-1:  # Bottom lid boundary
                a[f, f] += g[i] * h_wall[i] * A_cs
                b[f] += h_wall[i] * A_cs * T_lid_outlet

            # Solid temperature equation
            a[s, s] = (
                    (1 - eps) * V * rho_s * alpha1[i] / dt
                    + h_v[i] * V  # Heat transfer from fluid
            )
            a[s, f] = -h_v[i] * V * g[i]  # Heat transfer from fluid

            if i != 0:  # Inlet boundary
                a[s, s] += k_eff_intf[i-1] * A_cs / dz
                a[s, s-1] = -k_eff_intf[i-1] * A_cs / dz
            if i != n-1:  # Outlet boundary
                a[s, s] += k_eff_intf[i] * A_cs / dz
                a[s, s+1] = -k_eff_intf[i] * A_cs / dz

            b[s] = (1 - eps) * V * rho_s * (e_s_0[i] - alpha2[i]) / dt  # Previous time step

        x = np.linalg.solve(a, b)
        return x[:n], x[n:]

    @staticmethod
    @njit(parallel=True)
    def solve_lid_temperature(T_lid, T_f, T_env, h_wall, k, rho, cp, z, V, A, dt, *, reverse=False):
        """
        Solves for the lid temperature profile for the next time step. The boundary conditions are
        taken as:

        - Heat loss to environment by conduction at `i=0`
        - Heat loss to fluid by convection at `i=m-1`
        - Insulated at radial surfaces

        The `reverse` keyword can be used to reverse the axial orientation of the boundary conditions.

        Parameters:
             T_lid: Lid temperature for the previous time step [K].
             T_f: Fluid temperature estimate for the next time step at convection boundary [K].
             T_env: Environment temperature [K].
             h_wall: Wall heat transfer coefficient [W/m^2^⋅K].
             k: Thermal conductivity [W/m⋅K].
             rho: Density [kg/m^3^].
             cp: Specific heat capacity [J/kg⋅K].
             z: Axial positions of node centers [m].
             V: Node volumes [m^3^].
             A: Area of lid [m^2^].
             dt: Time step [s].
             reverse: Flag for reversing boundary conditions.
        """
        # Matrix setup
        m = T_lid.size
        a = np.zeros((m, m))
        b = np.zeros(m)

        # Reverse boundary conditions
        if reverse:
            T_lid = T_lid[::-1]
            k = k[::-1]
            rho = rho[::-1]
            cp = cp[::-1]
            z = z[::-1]
            V = V[::-1]

        # Calculate thermal conductivity at interfaces
        k_intf = 2 * k[:-1] * k[1:] / (k[:-1] + k[1:])

        # Fill matrix
        for i in prange(m):
            a[i, i] = V[i] * rho[i] * cp[i] / dt  # next time step
            b[i] = V[i] * rho[i] * cp[i] / dt * T_lid[i]  # previous time step

            if i == 0:  # Exterior lid node with heat transfer to environment
                a[i, i] += 2 * k[0] * A ** 2 / V[0]
                b[i] += 2 * k[0] * A ** 2 / V[0] * T_env
            else:
                a[i, i] += k_intf[i-1] * A / np.abs(z[i] - z[i-1])
                a[i, i-1] = -k_intf[i-1] * A / np.abs(z[i] - z[i-1])

            if i == m-1:  # Interior lid node with heat transfer to fluid
                a[i, i] += h_wall * A
                b[i] += h_wall * A * T_f
            else:
                a[i, i] += k_intf[i] * A / np.abs(z[i+1] - z[i])
                a[i, i+1] = -k_intf[i] * A / np.abs(z[i+1] - z[i])

        # Solve
        T = np.linalg.solve(a, b)
        return T[::-1] if reverse else T

    @staticmethod
    @njit(parallel=True)
    def _setup_wall_temperature_matrix(T_wall, T_f, T_env, h_wall, k, rho, cp, r, dr, dz, V, A_r, A_z, dt):
        """
        Sets up the diagonals of the sparse wall temperature matrix and the right hand side column vector.
        """
        n, m = T_wall.shape
        b = np.zeros(m * n)

        center = np.zeros(n * m)
        backward = np.zeros(n * m - 1)  # -z
        forward = np.zeros(n * m - 1)  # +z
        internal = np.zeros(n * (m - 1))  # -r
        external = np.zeros(n * (m - 1))  # +r

        # Calculate thermal conductivity at interfaces using harmonic mean
        k_intf = (dr[:-1] + dr[1:]) / (dr[:-1] / k[:-1] + dr[1:] / k[1:])

        # Fill matrix
        for i in prange(n):
            for j in prange(m):
                x = j * n + i  # Coordinate conversion

                center[x] = V[j] * rho[j] * cp[j] / dt
                b[x] = V[j] * rho[j] * cp[j] / dt * T_wall[i, j]  # previous time step

                # Radial boundary conditions
                if j == 0:  # Interior wall node with heat transfer to fluid
                    center[x] += h_wall[i] * A_r[0]
                    b[x] += h_wall[i] * A_r[0] * T_f[i]
                else:
                    center[x] += k_intf[j - 1] * A_r[j] / (r[j] - r[j - 1])
                    internal[x - n] = -k_intf[j - 1] * A_r[j] / (r[j] - r[j - 1])

                if j == m - 1:  # Exterior wall node with heat transfer to environment
                    center[x] += 2 * k[j] * A_r[j + 1] / dr[j]
                    b[x] += 2 * k[j] * A_r[j + 1] / dr[j] * T_env
                else:
                    center[x] += k_intf[j] * A_r[j + 1] / (r[j + 1] - r[j])
                    external[x] = -k_intf[j] * A_r[j + 1] / (r[j + 1] - r[j])

                # Axial boundary conditions
                if i != 0:
                    center[x] += k[j] * A_z[j] / dz
                    backward[x - 1] = -k[j] * A_z[j] / dz

                if i != n - 1:
                    center[x] += k[j] * A_z[j] / dz
                    forward[x] = -k[j] * A_z[j] / dz

        return [internal, backward, center, forward, external], b

    @staticmethod
    def solve_wall_temperature(T_wall, T_f, T_env, h_wall, k, rho, cp, r, dr, dz, V, A_r, A_z, dt):
        """
        Solves for the lid temperature profile for the next time step. The boundary conditions are
        taken as:

        - Heat loss to fluid by convection at `j=0`
        - Heat loss to environment by conduction at `j=m-1`
        - Insulated axially at `i=0` and `i=n-1`

        where the domain is discretized with `n` nodes in the axial direction and `m` in the radial
        direction, and the wall temperature `T_wall` has the shape `(n, m)`.

        Parameters:
             T_wall: Wall temperature for the previous time step [K].
             T_f: Fluid temperature estimate for the next time step [K].
             T_env: Environment temperature [K].
             h_wall: Wall heat transfer coefficient [W/m^2^⋅K].
             k: Thermal conductivity [W/m K].
             rho: Density [kg/m^3^].
             cp: Specific heat capacity [J/kg⋅K].
             r: Radial positions of node centers [m].
             dr: Radial node width [m].
             dz: Axial node spacing [m].
             V: Node volumes [m^3^].
             A_r: Area of node interfaces in the radial direction [m^2^].
             A_z: Area of node interfaces in the axial direction [m^2^].
             dt: Time step [s].
        """
        n, m = T_wall.shape

        diags, b = PackedBed._setup_wall_temperature_matrix(
            T_wall, T_f, T_env, h_wall, k, rho, cp, r, dr, dz, V, A_r, A_z, dt)
        A_sparse = scipy.sparse.diags(diags, offsets=[-n, -1, 0, 1, n], format="csr")
        return scipy.sparse.linalg.spsolve(A_sparse, b).reshape((m, n)).T

    def calculate_heat_transfer_coeffs(self, m_dot, T_f, k_f, cp_f, mu_f, k_s, E_s):
        """
        Calculates the relevant heat transfer coefficients for the packed bed.

        Parameters:
            m_dot: Mass flow rate [kg/s].
            T_f: Temperature of the fluid [K].
            k_f: Thermal conductivity of the fluid [W/m⋅K].
            cp_f: Specific heat capacity of the fluid [J/kg⋅K].
            mu_f: Dynamic viscosity of the fluid [Pa⋅s].
            k_s: Thermal conductivity of the solid [W/m⋅K].
            E_s: Emissivity of the solid.

        Returns:
            k_eff: Effective thermal conductivity [W/m⋅K].
            h_wall: Wall heat transfer coefficient [W/m^2^⋅K].
            h_v: Volumetric heat transfer coefficient [W/m^3^⋅K].
        """
        h_v = self.volumetric_convective_heat_transfer_coeff(m_dot, k_f, cp_f, self.eps, self.d, self.D)
        h_rv = self.void_radiative_heat_transfer_coeff(T_f, self.eps, E_s)
        h_rs = self.surface_radiative_heat_transfer_coeff(T_f, E_s)
        phi = self.effective_film_thickness_ratio(k_f, k_s, self.eps)
        k_eff = self.effective_thermal_conductivity(k_f, k_s, self.eps, h_rv, h_rs, phi, self.d)
        h_wall = (
            self.conv_wall_heat_transfer_coeff(m_dot, k_f, cp_f, mu_f, self.d, self.D)
            + self.cond_rad_wall_heat_transfer_coeff(k_f, k_s, h_rv, h_rs, self.eps, self.d, phi)
        )

        return k_eff, h_wall, h_v

    def calculate_fluid_enthalpy(self, T_f, P):
        """
        Returns the [`fluid`][sco2es.PackedBed.fluid] mass-specific enthalpy at each node.

        Parameters:
            T_f: Temperature [K].
            P: Pressure [Pa].
        """
        T_f = np.asarray(T_f)
        P = np.asarray(P)
        i_f = np.empty_like(T_f)
        for i in range(T_f.size):
            self.fluid.update(CP.PT_INPUTS, P[i], T_f[i])
            i_f[i] = self.fluid.hmass()
        return i_f

    def calculate_fluid_props(self, i_f, P):
        """
        Returns the [`fluid`][sco2es.PackedBed.fluid] temperature, thermal conductivity, density, viscosity, and
        specific heat capacity at each node.

        Parameters:
            i_f: Mass-specific enthalpy of the fluid [J/kg].
            P: Pressure [Pa].

        Returns:
            T_f: Temperature of the fluid [K].
            k_f: Thermal conductivity of the fluid [W/m⋅K].
            rho_f: Density of the fluid [kg/m^3^].
            mu_f: Dynamic viscosity of the fluid [Pa⋅s].
            cp_f: Specific heat capacity of the fluid [J/kg⋅K].
        """
        T_f = np.empty_like(i_f)
        k_f = np.empty_like(i_f)
        rho_f = np.empty_like(i_f)
        mu_f = np.empty_like(i_f)
        cp_f = np.empty_like(i_f)
        for i in range(i_f.size):
            self.fluid.update(CP.HmassP_INPUTS, i_f[i], P[i])
            T_f[i] = self.fluid.T()
            k_f[i] = self.fluid.conductivity()
            rho_f[i] = self.fluid.rhomass()
            mu_f[i] = self.fluid.viscosity()
            cp_f[i] = self.fluid.cpmass()
        return T_f, k_f, rho_f, mu_f, cp_f

    @staticmethod
    def biot_number(h_v, d, eps, k_s):
        r"""
        Calculates the biot number ($Bi$) for a packed bed[^1]

        $$
        Bi = \frac{h_v d^2}{36(1 - \varepsilon) k_s}
        $$

        [^1]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy
        storage systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal
        Engineering, vol. 183, p. 116032, 2021.

        Parameters:
            h_v: Volumetric heat transfer coefficient, $h_v$ [W/m^3^⋅K].
            d: Particle diameter, $d$ [m].
            eps: Void fraction, $\varepsilon$.
            k_s: Thermal conductivity of the solid, $k_s$ [W/m⋅K].
        """
        return h_v * d ** 2 / (36 * (1 - eps) * k_s)

    @staticmethod
    def effective_film_thickness_ratio(k_f, k_s, eps):
        r"""
        Calculates the ratio between the effective thickness of the fluid film adjacent to the surface of two solid
        particles and the particle diameter according to the interpolation of Kunii and Smith[^1] for
        $0.260 \leq \varepsilon \leq 0.476$

        $$
        \phi = \phi_2 + (\phi_1 - \phi_2) \frac{\varepsilon - 0.260}{0.216}
        $$

        where

        $$
        \phi_i = \frac{1}{2} \frac{\left(\frac{\kappa-1}{\kappa}\right)^2 \sin^2{\theta_i}} {\ln\left[\kappa -
        (\kappa - 1)\cos{\theta_i}\right] - \left(\frac{\kappa - 1}{\kappa}\right) (1 - \cos{\theta_i)}} - \frac{2}{3\kappa}
        $$

        where $\kappa = k_s/k_f$ and the interpolation bounds are given by $\sin^2 \theta_i = \frac{1}{n_i}$ for
        $n_1 = 1.5$ and $n_2 = 4\sqrt{3}$.

        [^1]: D. Kunii and J. Smith, “Heat transfer characteristics of porous rocks,” AIChE Journal, vol. 6, no. 1, pp.
        71–78, 1960.

        Parameters:
            k_s: Thermal conductivity of the solid, $k_s$ [W/m⋅K].
            k_f: Thermal conductivity of the fluid, $k_f$ [W/m⋅K].
            eps: Void fraction, $\varepsilon$.
        """
        kappa = k_s / k_f

        def phi_i(n_i):
            cos_theta_i = (1 - 1 / n_i) ** 0.5
            return 1 / 2 * ((kappa - 1) / kappa) ** 2 / n_i / (
                    np.log(kappa - (kappa - 1) * cos_theta_i) - (kappa - 1) / kappa * (1 - cos_theta_i)
            ) - 2 / (3 * kappa)

        phi1 = phi_i(1.5)
        phi2 = phi_i(4 * 3**0.5)
        return phi1 + (phi2 - phi1) * (eps - 0.26) / 0.216

    @staticmethod
    def void_radiative_heat_transfer_coeff(T, eps, E_s):
        r"""
        Calculates the void-to-void radiative heat transfer coefficient for a packed bed given by the correlation of Yagi
        and Kunii[^1]

        $$
        h_{rv} = \frac{0.1952}{1 + \frac{\varepsilon (1-E_s)}{2E_s (1-\varepsilon)}} \left(\frac{T}{100}\right)^3
        $$

        [^1]: S. Yagi and D. Kunii, “Studies on effective thermal conductivities in packed beds,” AIChE Journal,
        vol. 3, no. 3, pp. 373–381, 1957.

        Parameters:
            T: Temperature, $T$ [K].
            eps: Void fraction, $\varepsilon$.
            E_s: Emissivity of the solid, $E_s$.
        """
        return 0.1952 * (T/100)**3 / (1 + eps * (1 - E_s) / (2 * E_s * (1 - eps)))

    @staticmethod
    def surface_radiative_heat_transfer_coeff(T, E_s):
        r"""
        Calculates the surface-to-surface radiative heat transfer coefficient for a packed bed given by the correlation of
        Yagi and Kunii[^1]

        $$
        h_{rs} = \frac{0.1952E_s}{2-E_s} \left( \frac{T}{100}\right)^3
        $$

        [^1]: S. Yagi and D. Kunii, “Studies on effective thermal conductivities in packed beds,” AIChE Journal,
        vol. 3, no. 3, pp. 373–381, 1957.

        Parameters:
            T: Temperature, $T$ [K].
            E_s: Emissivity of the solid, $E_s$.

        """
        return 0.1952 * (T/100)**3 * E_s / (2 - E_s)

    @staticmethod
    def effective_thermal_conductivity(k_f, k_s, eps, h_rv, h_rs, phi, d, *, beta=0.9):
        r"""
        Approximates the effective thermal conductivity in a packed bed using the model of Kunii and Smith[^1]

        $$
        k_{eff} = k_f \left[\varepsilon\left(1+\beta \frac{h_{r\nu}d}{k_f}\right) +
        \frac{\beta (1-\varepsilon}{\frac{1}{\frac{1}{\phi}+\frac{h_{rs}d}{k_f}}
        +\gamma\left(\frac{k_f}{k_s}\right)} \right]
        $$

        [^1]: D. Kunii and J. Smith, “Heat transfer characteristics of porous rocks,” AIChE Journal, vol. 6, no. 1,
        pp. 71–78, 1960.

        Parameters:
            k_f: Thermal conductivity of the fluid, $k_f$ [W/m⋅K].
            k_s: Thermal conductivity of the solid, $k_s$ [W/m⋅K].
            eps: Void fraction. $\varepsilon$.
            h_rv: Void-to-void radiative heat transfer coefficient, $h_{rv}$ [W/m^2^⋅K].
            h_rs: Surface-to-surface radiative heat transfer coefficient, $h_{rs}$ [W/m^2^⋅K].
            phi: Effective film thickness ratio, $\phi$.
            d: Solid particle diameter, $d$ [m].
            beta: Ratio of the average length between the centers of two neighboring solids to
                the mean particle diameter, $\beta = 0.9$ (default).
        """
        return k_f * (
            eps * (1 + beta * h_rv * d / k_f) +
            beta * (1 - eps) / (
                    1 / (1 / phi + h_rs * d / k_f)
                    + 2/3 * k_f / k_s)
        )

    @staticmethod
    @np.vectorize
    def volumetric_convective_heat_transfer_coeff(m_dot, k_f, cp_f, eps, d, D):
        r"""
        Calculates the volumetric convective heat transfer coefficient as the product of the convective heat
        transfer coefficient for a spherical particle and the ratio between the particles total heat transfer area
        and the total volume

        $$
        h_v = h_{part} \frac{6(1-\varepsilon)}{d}
        $$

        The particle convective heat transfer coefficient is given by Pfeffer's correlation[^1]

        $$
        h_{part} = 1.26 \left[\frac{1-(1-\varepsilon)^{5/3}}{W}\right]^{1/3} (c_{p_f} G)^{1/3}
        \left(\frac{k_f}{d}\right)^{2/3}
        $$

        where $W$ is given by

        $$
        W = 2-3(1-\varepsilon)^{1/3}+3(1-\varepsilon)^{5/3}-2(1-\varepsilon)^2
        $$

        and $G$ is the effective mass flow rate per unit area given by

        $$
        G = \frac{4\dot{m}}{\varepsilon \pi D^2}
        $$

        The lower limit of analytical heat transfer is given by

        $$
        h_{part} = \frac{2k_f}{d}
        $$

        for a heated isothermal sphere in a quiescent fluid medium; therefore, the maximum of the two values is taken.

        [^1]: R. Pfeffer, “Heat and mass transport in multiparticle systems,” Industrial & Engineering Chemistry
        Fundamentals, vol. 3, no. 4, pp. 380–383, 1964.

        Parameters:
            m_dot: Mass flow rate [kg/s].
            k_f: Thermal conductivity of the fluid, $k_f$ [W/m⋅K].
            cp_f: Specific heat capacity of the fluid, $c_{p_f}$ [J/kg⋅K].
            eps: Void fraction, $\varepsilon$.
            d: Particle diameter, $d$ [m].
            D: Diameter [m].
        """
        G = 4 * m_dot / (eps * np.pi * D**2)
        W = 2 - 3 * (1 - eps)**(1/3) + 3 * (1 - eps)**(5/3) - 2 * (1 - eps)**2
        h_part = np.max([
            1.26 * ((1 - (1 - eps)**(5/3)) / W)**(1/3) * (cp_f * G)**(1/3) * (k_f / d)**(2/3),  # Pfeffer's correlation
            2 * k_f / d  # Lower limit
        ])
        return h_part * 6 * (1 - eps) / d

    @staticmethod
    def conv_wall_heat_transfer_coeff(m_dot, k_f, cp_f, mu_f, d, D):
        r"""
        Returns the convective heat transfer coefficient between the fluid and the wall

        $$
        h_{wall}^{cv} = \left( 2.58 Re_d^{1/3} Pr^{1/3} + 0.094 Re_d^{0.8} Pr^{0.4} \right) \frac{k_f}{d}
        $$

        according to the correlation of Beek[^1]. The Reynolds number $Re_d$ is defined as

        $$
        Re_d = \frac{\rho_f u_0 d}{\mu_f}
        $$

        where $u_0$ is the superficial velocity, or the velocity if no particles were present, given by

        $$
        u_0 = \frac{\dot{m}}{\rho_f A}
        $$

        The Prandtl number $Pr$ is defined as

        $$
        Pr = \frac{{c_p}_f \mu_f}{k_f}
        $$

        [^1]: J. Beek, “Design of packed catalytic reactors,” in Advances in Chemical Engineering, Elsevier,
        1962, pp. 203–271.

        Parameters:
            m_dot: Mass flow rate, $\dot{m}$ [kg/s].
            k_f: Thermal conductivity of the fluid, $k_f$ [W/m⋅K].
            cp_f: Specific heat capacity of the fluid, ${c_p}_f$ [J/kg⋅K].
            mu_f: Dynamic viscosity of the fluid, $\mu_f$ [Pa⋅s].
            d: Particle diameter, $d$ [m].
            D: Diameter, $D$ [m].
        """
        Re_d = d * m_dot / (mu_f * np.pi * D**2 / 4)
        Pr = cp_f * mu_f / k_f

        return (2.58 * Re_d**(1/3) * Pr**(1/3) + 0.094 * Re_d**0.8 * Pr**0.4) * k_f / d

    @staticmethod
    def cond_rad_wall_heat_transfer_coeff(k_f, k_s, h_rv, h_rs, eps, d, phi):
        r"""
        Calculates the conductive and radiative wall heat transfer coefficient according to Ofuchi and Kunii[^1]

        $$
        h_{wall}^{cd,ra} = \frac{k_{eff}^{stag} k_{wall}^{stag}}{k_{eff}^{stag} - \frac{k_{wall}^{stag}}{2}}
        $$

        where

        $$
        k_{eff}^{stag} = k_f \left[\varepsilon \left(1 + \frac{h_{rv}d}{k_f}\right) +
        \frac{1-\varepsilon}{\left(\frac{1}{\phi}+\frac{h_{rs}d}{k_f}\right)^{-1} + \frac{2}{3\kappa}}\right]
        $$

        and

        $$
        k_{wall}^{stag} = k_f \left[\varepsilon_{wall} \left(2 + \frac{h_{rv}d}{k_f}\right) +
        \frac{1-\varepsilon_{wall}}{\left(\frac{1}{\phi_{wall}}+\frac{h_{rs}d}{k_f}\right)^{-1} + \frac{1}{3\kappa}}\right]
        $$

        where the wall porosity, $\varepsilon_{wall}$, is assumed to be $0.4$ and $\phi_{wall}$ is given by

        $$
        \phi_{wall} = \frac{1}{4} \frac{\left(\frac{\kappa-1}{\kappa}\right)^2}{\ln{\kappa} - \frac{\kappa-1}{\kappa}} -
        \frac{1}{3\kappa}
        $$

        where $\kappa = k_s/k_f$.

        [^1]: K. Ofuchi and D. Kunii, “Heat-transfer characteristics of packed beds with stagnant fluids,” International
        Journal of Heat and Mass Transfer, vol. 8, no. 5, pp. 749–757, 1965.

        Parameters:
            k_f: Thermal conductivity of the fluid, $k_f$ [W/m⋅K].
            k_s: Thermal conductivity of the solid, $k_s$ [W/m⋅K].
            h_rv: void-to-void radiative heat transfer coefficient, $h_{rv}$ [W/m^2^⋅K].
            h_rs: Solid-to-solid radiative heat transfer coefficient, $h_{rs}$ [W/m^2^⋅K].
            eps: Void fraction, $\varepsilon$.
            d: Particle diameter, $d$ [m].
            phi: Effective film thickness ratio, $\phi$.
        """
        kappa = k_s / k_f
        eps_wall = 0.4

        k_stag_eff = k_f * (
            eps * (1 + h_rv * d / k_f)
            + (1 - eps) / ((1/phi + h_rs * d / k_f)**-1 + 2 / (3 * kappa))
        )

        phi_wall = 1/4 * ((kappa - 1) / kappa)**2 / (np.log(kappa) - (kappa - 1) / kappa) - 1/(3 * kappa)
        k_stag_wall = k_f * (
            eps_wall * (2 + h_rv * d / k_f)
            + (1 - eps_wall) / ((1/phi_wall + h_rs * d / k_f)**-1 + 1/(3 * kappa))
        )

        return k_stag_eff * k_stag_wall / (k_stag_eff - k_stag_wall / 2)

    @staticmethod
    @np.vectorize
    def pressure_drop(
            dz: float,
            rho_f: float,
            mu_f: float,
            G: float,
            eps: float,
            d: float,
            *,
            psi: float = 0.9,
            xi1: float = 180,
            xi2: float = 1.8
    ):
        r"""
        Calculates the pressure drop using the modified Ergun's equation[^1]

        $$
        \Delta P = \frac{\Delta z G^2}{\rho_f d} \left[ \xi_1 \frac{(1-\epsilon)^2}{\epsilon^3 \psi^2} \frac{\mu_f}{Gd}
        + \xi_2 \frac{(1-\epsilon)}{\epsilon^3 \psi}\right]
        $$

        [^1]: I. Macdonald, M. El-Sayed, K. Mow, and F. Dullien, “Flow through porous media-the Ergun equation revisited,”
        Industrial & Engineering Chemistry Fundamentals, vol. 18, no. 3, pp. 199–208, 1979.

        Parameters:
            dz: Axial position step size, $\Delta z$ [m].
            rho_f: Density of the fluid, $\rho_f$ [kg/m^3^].
            mu_f: Dynamic viscosity of the fluid, $\mu_f$ [Pa⋅s].
            G: Effective mass flow rate per unit cross-section, $G$ [kg/m^2^⋅s].
            eps: Void fraction, $\varepsilon$.
            d: Particle diameter, $d$ [m].
            psi: Sphericity, $\psi$.
            xi1: Viscous loss coefficient, $\xi_1$.
            xi2: Inertial loss coefficient, $\xi_2$.

        """
        return dz * G**2 / (rho_f * d) * (
            xi1 * (1 - eps)**2 * mu_f / (eps**3 * psi**2 * G * d)  # Viscous loss
            + xi2 * (1 - eps) / (eps**3 * psi)  # Inertial loss
        ) if G != 0 else 0
