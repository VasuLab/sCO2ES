import numpy as np
import numpy.typing as npt
from numba import njit, prange
from matplotlib import pyplot as plt
import CoolProp as CP


class PackedBedModel:
    r"""
    An implementation of the packed bed thermal energy storage model described by Battisti et al.[^1].

    !!! Warning "Assumptions"

        - Constant solid phase density
        - Constant wall thermal properties
        - Constant axial spacing

    Attributes:
        n: Number of nodes in the axial direction.
        m: Number of wall or lid nodes in the radial or axial directions, respectively.
        nodes: Total number of nodes (`2(n+m) + nm`).
        r_wall (m): Radial positions of wall nodes from centerline [m].
        r_bound (m + 1): Radial positions of wall cell boundaries from centerline [m].
        z_top_lid (m): Axial positions of top lid nodes from charging inlet in direction of flow [m].
        z_bottom_lid (m): Axial positions of bottom lid nodes from charging inlet in direction of flow [m].
        V_wall (m): Volume of the wall cells [m^3^].
        A_wall_z (m): Surface area of the wall cell boundary in the axial direction [m^2^].
        A_wall_r (m+1): Surface area of the wall cell boundary in the radial direction [m^2^].

    [^1]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy storage
    systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal Engineering, vol.
    183, p. 116032, 2021.
    """

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

    max_iter: int = 100
    """Maximum number of iterations for the loop."""

    def __init__(
            self,
            T_d: float,
            P: float,
            L: float,
            D: float,
            d: float,
            eps: float,
            rho_s: float,
            T_env: float,
            t_wall: npt.ArrayLike,
            k_wall: npt.ArrayLike,
            rho_wall: npt.ArrayLike,
            cp_wall: npt.ArrayLike,
            *,
            n: int = 100,
            n_wall: int | npt.ArrayLike = 10):
        """
        Parameters:
            T_d: Discharge temperature [K].
            P: Initial pressure [Pa].
            L: Domain length [m].
            D: Internal tank diameter [m].
            d: Particle diameter [m].
            eps: Void fraction.
            rho_s: Density of the solid [kg/m^3^].
            T_env: Temperature of the environment [K].
            t_wall: Thickness of each wall layer [m].
            k_wall: Thermal conductivity of each wall layer [W/m⋅K].
            rho_wall: Density of each wall layer [kg/m^3^].
            cp_wall: Specific heat capacity of each wall layer [J/kg⋅K].
            n: Number of axial nodes.
            n_wall: Number of nodes for each wall layer.
        """

        # Simulation parameters
        self.n = n
        self.L = L
        self.dz = L / n
        self.t = np.array([0])
        self.z = np.linspace(self.dz / 2, L - self.dz / 2, n)

        # Packed bed parameters
        self.D = D
        self.d = d
        self.eps = eps

        # State variables
        self.P = np.full((1, n), P)
        self.T_f = np.full((1, n), T_d, dtype=float)
        self.T_s = np.full((1, n), T_d, dtype=float)

        # Property variables
        self.k_f, self.rho_f, self.mu_f, self.cp_f = self.calculate_fluid_props(self.T_f[0], self.P[0])
        self.i_f = ...
        self.rho_s = rho_s
        self.E_s, self.k_s = self.calculate_solid_props(self.T_s[0])

        # Field variables
        self.m_dot = np.zeros(n)  # Initially stationary
        self.k_eff, self.h_wall, self.h_v = self.calculate_heat_transfer_coeffs(
            self.m_dot, self.T_f[0], self.k_f, self.cp_f, self.mu_f, self.k_s, self.E_s)

        # Wall/lid parameters
        self.T_env = T_env

        t_wall = np.asarray(t_wall, dtype=float)
        k_wall = np.asarray(k_wall, dtype=float)
        rho_wall = np.asarray(rho_wall, dtype=float)
        cp_wall = np.asarray(cp_wall, dtype=float)

        assert t_wall.size == k_wall.size == rho_wall.size == cp_wall.size
        if isinstance(n_wall, int):
            n_wall = np.full(t_wall.shape, n_wall)

        self.m = sum(n_wall)
        self.nodes = 2 * (n + self.m) + n * self.m
        self.dr = np.repeat(t_wall / n_wall, n_wall)

        # Determine wall grid properties
        dx_bound = np.add.accumulate(np.insert(self.dr, 0, 0))
        dx_center = (dx_bound[1:] + dx_bound[:-1]) / 2

        self.r_bound = self.D / 2 + dx_bound
        self.r_wall = self.D / 2 + dx_center
        self.z_top_lid = -dx_center[::-1]
        self.z_bottom_lid = L + dx_center

        self.A_wall_z = np.pi * (self.r_bound[1:]**2 - self.r_bound[:-1]**2)
        self.V_wall = self.A_wall_z * self.dz
        self.A_wall_r = 2 * np.pi * self.dz * self.r_bound

        self.A_lid = np.pi * self.D ** 2 / 4
        self.V_bottom_lid = self.A_lid * np.diff(dx_bound)
        self.V_top_lid = self.V_bottom_lid[::-1]

        # Initialize wall/lid temperature distribution
        self.T_wall = np.empty((1, self.n, self.m), dtype=float)
        self.T_top_lid = np.empty((1, self.m), dtype=float)
        self.T_bottom_lid = np.empty((1, self.m), dtype=float)

        # Calculate heat flux through lid and walls
        R_lid = t_wall / k_wall  # Convection/conduction resistance
        Q_lid = (T_d - T_env) / np.sum(R_lid)  # Heat flux through the lid
        T_lid_bound = np.array([T_d, *(T_d - np.add.accumulate(Q_lid * R_lid))])

        self.r_layer_bound = D/2 + np.array([0, *np.add.accumulate(t_wall)])
        R_wall = np.log(self.r_layer_bound[1:] / self.r_layer_bound[:-1]) / (2 * np.pi * k_wall * L)
        Q_wall = (T_d - T_env) / np.sum(R_wall)
        T_wall_bound = [T_d, *(T_d - np.add.accumulate(Q_wall * R_wall))]

        # Calculate coordinates and temperature profiles for walls and lids
        T_wall = []
        T_lid = []

        for i in range(n_wall.size):
            dx = t_wall[i] / (2 * n_wall[i]) * (1 + 2 * np.arange(n_wall[i]))
            T_wall.append(T_wall_bound[i] -
                          Q_wall * np.log(1 + dx / self.r_layer_bound[i]) / (2 * np.pi * L * k_wall[i]))
            T_lid.append(T_lid_bound[i] + np.diff(T_lid_bound)[i] * dx / t_wall[i])

        T_lid = np.array(T_lid).flatten()
        T_wall = np.array(T_wall).flatten()
        self.T_top_lid[0] = T_lid[::-1]
        self.T_bottom_lid[0] = T_lid
        self.T_wall[0] = T_wall

        # Fill wall/lid properties
        self.k_wall = np.repeat(k_wall, n_wall)
        self.rho_wall = np.repeat(rho_wall, n_wall)
        self.cp_wall = np.repeat(cp_wall, n_wall)

        self.k_bottom_lid = self.k_wall
        self.rho_bottom_lid = self.rho_wall
        self.cp_bottom_lid = self.cp_wall

        self.k_top_lid = self.k_wall[::-1]
        self.rho_top_lid = self.rho_wall[::-1]
        self.cp_top_lid = self.cp_wall[::-1]

    def advance(self, t):
        """
        The main solver loop.
        """

        while not self.stop():
            self.step()

            Bi = self.biot_number(self.h_v, self.d, self.eps, self.k_s)
            if not np.all(Bi <= 0.1):
                raise Exception("Biot number exceeded acceptable threshold.")

    def stop(self) -> bool:
        """A function that indicates when the solver should stop."""
        ...

    def step(self, dt):
        """
        Calculates the state of the packed bed at the next time step using an iterative algorithm.

        Returns:
            Number of iterations required for convergence.

        """

        # Next iteration state arrays
        P = np.copy(self.P[-1])
        T_f = np.copy(self.T_f[-1])
        T_s = np.copy(self.T_s[-1])
        T_wall = np.copy(self.T_wall[-1])
        T_top_lid = np.copy(self.T_top_lid[-1])
        T_bottom_lid = np.copy(self.T_bottom_lid[-1])
        i_f = np.copy(self.i_f)
        rho_f = np.copy(self.rho_f)
        m_dot = np.copy(self.m_dot)
        h_v = np.copy(self.h_v)
        h_wall = np.copy(self.h_wall)

        for b in range(self.max_iter):
            # Previous iteration state arrays
            P_prev = np.copy(P)
            T_f_prev = np.copy(T_f)
            T_s_prev = np.copy(T_s)
            T_wall_prev = np.copy(T_wall)
            T_top_lid_prev = np.copy(T_top_lid)
            T_bottom_lid_prev = np.copy(T_bottom_lid)
            i_f_prev = np.copy(i_f)
            rho_f_prev = np.copy(rho_f)
            m_dot_prev = np.copy(m_dot)
            h_v_prev = np.copy(h_v)
            h_wall_prev = np.copy(h_wall)

            # Update thermodynamic properties
            k_f, rho_f, mu_f, cp_f = self.calculate_fluid_props(T_f, P)
            E_s, k_s = self.calculate_solid_props(T_s)
            k_eff, h_wall, h_v = self.calculate_heat_transfer_coeffs(m_dot, T_f, k_f, cp_f, mu_f, k_s, E_s)

            # Solve for lid/wall temperatures
            T_top_lid = self.solve_lid_temperature(
                self.T_top_lid[-1], T_f[0], self.T_env, h_wall[0],
                self.k_top_lid, self.rho_top_lid, self.cp_top_lid,
                self.z_top_lid, self.V_top_lid, self.A_lid, dt,
                reverse=False
            )

            T_bottom_lid = self.solve_lid_temperature(
                self.T_bottom_lid[-1], T_f[-1], self.T_env, h_wall[-1],
                self.k_bottom_lid, self.rho_bottom_lid, self.cp_bottom_lid,
                self.z_bottom_lid, self.V_bottom_lid, self.A_lid, dt,
                reverse=True
            )

            T_wall = self.solve_wall_temperature(
                self.T_wall[-1], T_f, self.T_env, h_wall,
                self.k_wall, self.rho_wall, self.cp_wall,
                self.r_wall, self.dr, self.dz,
                self.V_wall, self.A_wall_r, self.A_wall_z, dt
            )

            # Check convergence
            converged = np.all([
                # np.all(np.abs(P - P_prev) <= self.atol_P),
                # np.all(np.abs(T_f - T_f_prev) <= self.atol_T_f),
                # np.all(np.abs(T_s - T_s_prev) <= self.atol_T_s),
                np.all(np.abs(T_wall - T_wall_prev) <= self.rtol_T_wall * T_wall),
                np.all(np.abs(T_top_lid - T_top_lid_prev) <= self.rtol_T_wall * T_top_lid),
                np.all(np.abs(T_bottom_lid - T_bottom_lid_prev) <= self.rtol_T_wall * T_bottom_lid),
                # np.all(np.abs(i_f - i_f_prev) <= self.rtol_i_f * i_f),
                # np.all(np.abs(rho_f - rho_f_prev) <= self.rtol_rho_f * rho_f),
                # np.all(np.abs(m_dot - m_dot_prev) <= self.rtol_m_dot * m_dot),
                # np.all(np.abs(m_dot - m_dot_prev) <= self.rtol_m_dot * m_dot),
                # np.all(np.abs(h_v - h_v_prev) <= self.rtol_h * h_v),
                # np.all(np.abs(h_wall - h_wall_prev) <= self.rtol_h * h_wall),
            ])

            if converged:
                self.t = np.append(self.t, self.t[-1] + dt)
                self.P = np.append(self.P, [P], axis=0)
                self.T_f = np.append(self.T_f, [T_f], axis=0)
                self.T_s = np.append(self.T_s, [T_s], axis=0)
                self.T_wall = np.append(self.T_wall, [T_wall], axis=0)
                self.T_top_lid = np.append(self.T_top_lid, [T_top_lid], axis=0)
                self.T_bottom_lid = np.append(self.T_bottom_lid, [T_bottom_lid], axis=0)
                self.i_f = i_f
                self.rho_f = rho_f
                self.m_dot = m_dot
                self.h_v = h_v
                self.h_wall = h_wall

                return b + 1

        raise Exception("Maximum number of iterations reached without convergence.")

    @staticmethod
    @njit(parallel=True)
    def solve_lid_temperature(T_lid, T_f, T_env, h_wall, k, rho, cp, z, V, A, dt, *, reverse=False):
        """
        :material-lightning-bolt:{ .parallel } Parallelized

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
    def solve_wall_temperature(T_wall, T_f, T_env, h_wall, k, rho, cp, r, dr, dz, V, A_r, A_z, dt):
        """
        :material-lightning-bolt:{ .parallel } Parallelized

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
        # Matrix setup
        n, m = T_wall.shape
        a = np.zeros((m * n, m * n))
        b = np.zeros(m * n)

        # Calculate thermal conductivity at interfaces using harmonic mean
        k_intf = (dr[:-1] + dr[1:]) / (dr[:-1] / k[:-1] + dr[1:] / k[1:])

        # Fill matrix
        for i in prange(n):
            for j in prange(m):
                x = j * n + i  # Coordinate conversion

                a[x, x] = V[j] * rho[j] * cp[j] / dt  # next time step
                b[x] = V[j] * rho[j] * cp[j] / dt * T_wall[i, j]  # previous time step

                # Radial boundary conditions
                if j == 0:  # Interior wall node with heat transfer to fluid
                    a[x, x] += h_wall[i] * A_r[0]
                    b[x] += h_wall[i] * A_r[0] * T_f[i]
                else:
                    a[x, x] += k_intf[j-1] * A_r[j] / (r[j] - r[j - 1])
                    a[x, x-n] = -k_intf[j-1] * A_r[j] / (r[j] - r[j - 1])

                if j == m-1:  # Exterior wall node with heat transfer to environment
                    a[x, x] += 2 * k[j] * A_r[j + 1] / dr[j]
                    b[x] += 2 * k[j] * A_r[j + 1] / dr[j] * T_env
                else:
                    a[x, x] += k_intf[j] * A_r[j + 1] / (r[j + 1] - r[j])
                    a[x, x+n] = -k_intf[j] * A_r[j + 1] / (r[j + 1] - r[j])

                # Axial boundary conditions
                if i != 0:
                    a[x, x] += k[j] * A_z[j] / dz
                    a[x, x-1] = -k[j] * A_z[j] / dz

                if i != n-1:
                    a[x, x] += k[j] * A_z[j] / dz
                    a[x, x+1] = -k[j] * A_z[j] / dz

        # Solve
        return np.linalg.solve(a, b).reshape((m, n)).T

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

    @staticmethod
    def calculate_fluid_props(T_f, P):
        """
        Returns the thermal conductivity, density, viscosity, and specific heat capacity of CO~2~ at each node
        using [`CoolProp`](http://www.coolprop.org/).

        Parameters:
            T_f: Temperature of the fluid [K].
            P: Pressure [Pa].

        Returns:
            k_f: Thermal conductivity of the fluid [W/m⋅K].
            rho_f: Density of the fluid [kg/m^3^].
            mu_f: Dynamic viscosity of the fluid [Pa⋅s].
            cp_f: Specific heat capacity of the fluid [J/kg⋅K].
        """
        k_f = np.empty_like(T_f)
        rho_f = np.empty_like(T_f)
        mu_f = np.empty_like(T_f)
        cp_f = np.empty_like(T_f)
        for i in range(T_f.size):
            k_f[i] = CP.CoolProp.PropsSI("CONDUCTIVITY", "T", T_f[i], "P", P[i], "CO2")
            rho_f[i] = CP.CoolProp.PropsSI("DMASS", "T", T_f[i], "P", P[i], "CO2")
            mu_f[i] = CP.CoolProp.PropsSI("VISCOSITY", "T", T_f[i], "P", P[i], "CO2")
            cp_f[i] = CP.CoolProp.PropsSI("CPMASS", "T", T_f[i], "P", P[i], "CO2")
        return k_f, rho_f, mu_f, cp_f

    @staticmethod
    @njit(parallel=True)
    def calculate_solid_props(T_s):
        """
        Returns the temperature-dependent emissivity[^1] and thermal conductivity[^2] of alumina for each node.

        [^1]: M. E. Whitson Jr, "Handbook of the Infrared Optical Properties of Al2O3. Carbon, MGO and ZrO2. Volume 1,"
        El Segundo/CA, 1975.
        [^2]: "AETG/UC San Diego," [Online]. Available: www.ferp.ucsd.edu/LIB/PROPS/PANOS/al2o3.html

        Parameters:
            T_s: Temperature of the solid [K].

        Returns:
            E_s: Emissivity of the solid.
            k_s: Thermal conductivity of the solid [W/m⋅K].
        """
        T_star = (T_s - 953.8151) / 432.1046
        E_s = 0.5201 - 0.1794 * T_star + 0.01343 * T_star**2 + 0.01861 * T_star**3
        k_s = 85.686 - 0.22972 * T_s + 2.607e-4 * T_s**2 - 1.3607e-7 * T_s**3 + 2.7092e-11 * T_s**4
        return E_s, k_s

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
    def pressure_drop(dz, rho_f, mu_f, G, eps, d, psi, *, xi1: float = 180, xi2: float = 1.8):
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
        )

    def plot_temperature_contour(self, ax=None):
        """
        Plots the temperature contour of the solid phase and tank walls/lids on the given Matplotlib axes object,
        `ax` (if not given, a new figure is created).
        """
        if ax is None:
            _, ax = plt.subplots()

        r = np.array([*(-self.r_wall[::-1]), -self.D / 2, self.D / 2, *self.r_wall])
        z = np.array([*self.z_top_lid, *self.z, *self.z_bottom_lid])

        r, z = np.meshgrid(r, z)
        T = np.full_like(r, np.nan)

        # Fluid temperature
        wall_nodes = self.r_wall.size
        T[wall_nodes:-wall_nodes, wall_nodes:-wall_nodes] = np.tile(self.T_s[-1], (2, 1)).T

        # Wall temperature
        T[wall_nodes:-wall_nodes, -wall_nodes:] = self.T_wall[-1]
        T[wall_nodes:-wall_nodes, wall_nodes - 1::-1] = self.T_wall[-1]

        # Lid temperature
        T[:wall_nodes, wall_nodes:-wall_nodes] = np.tile(self.T_top_lid[-1], (2, 1)).T
        T[-wall_nodes:, wall_nodes:-wall_nodes] = np.tile(self.T_bottom_lid[-1], (2, 1)).T

        # Plot
        plt.contourf(r, z, T - 273, levels=100)

        # Draw boundaries
        plt.axvline(0, color="black", linestyle="--", alpha=0.75)

        plt.axvline(self.D / 2, color="black", alpha=0.75)
        plt.axvline(-self.D / 2, color="black", alpha=0.75)
        plt.axhline(0, color="black", alpha=0.75)
        plt.axhline(self.L, color="black", alpha=0.75)

        # Format plot
        plt.xlim(-self.r_wall.max(), self.r_wall.max())
        plt.ylim(self.z_bottom_lid.max(), self.z_top_lid.min())

        cbar = plt.colorbar()
        cbar.set_label("Temperature [°C]")
        ax.set_ylabel("Axial distance [m]")
        ax.set_xlabel("Radial distance [m]")

        ax.set_facecolor("black")

