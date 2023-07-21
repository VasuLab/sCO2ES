import numpy as np


def effective_thermal_conductivity(
        T_f: float,
        T_s: float,
        k_f: float,
        k_s: float,
        void: float,
        emiss: float,
        d: float,
        *,
        beta: float = 0.9,
):
    r"""
    Approximates the effective thermal conductivity in a packed bed using the model of Kunii and Smith[^1]:

    $$
    k_{eff} = k_f \left[\varepsilon\left(1+\beta \frac{h_{r\nu}d}{k_f}\right) +
    \frac{\beta (1-\varepsilon}{\frac{1}{\frac{1}{\phi}+\frac{h_{rs}d}{k_f}}
    +\gamma\left(\frac{k_f}{k_s}\right)} \right]
    $$

    The void-to-void and surface-to-surface radiative heat transfer coefficients are from Yagi and Kunii[^2].

    [^1]: D. Kunii and J. Smith, “Heat transfer characteristics of porous rocks,” AIChE Journal, vol. 6, no. 1,
    pp. 71–78, 1960.
    [^2]: S. Yagi and D. Kunii, “Studies on effective thermal conductivities in packed beds,” AIChE Journal,
    vol. 3, no. 3, pp. 373–381, 1957.

    Parameters:
        T_f: Fluid temperature [K].
        T_s: Solid temperature [K].
        k_f: Thermal conductivity of the fluid [W/mK].
        k_s: Thermal conductivity of the solid [W/mK].
        void: Void fraction [-].
        emiss: Emissivity of the solid [-].
        d: Solid particle diameter [m].
        beta: Ratio of the average length between the centers of two neighboring solids to
            the mean particle diameter.
    """

    gamma = 2 / 3

    h_rv = 0.1952 * (T_f/100)**3 / (1 + void * (1 - emiss) / (2 * emiss * (1 - void)))
    h_rs = 0.1952 * (T_s/100) ** 3 * emiss / (2 - emiss)

    kappa = k_s / k_f

    def phi_i(n_i):
        cos_theta_i = (1 - 1/n_i)**0.5
        return 1/2 * ((kappa - 1) / kappa)**2 / n_i / (
            np.log(kappa - (kappa - 1) * cos_theta_i) - (kappa - 1) / kappa * (1 - cos_theta_i)
        ) - 2 / (3 * kappa)

    phi1 = phi_i(1.5)
    phi2 = phi_i(4 * 3 ** 0.5)
    phi = phi2 + (phi1 - phi2) * (void - 0.26) / (0.476 - 0.26)

    return k_f * (
            void * (1 + beta * h_rv * d / k_f) +
            beta * (1 - void) / (
                    1 / (1 / phi + h_rs * d / k_f)
                    + gamma * k_f / k_s)
    )


def pressure_drop(dz, rho_f, mu_f, G, eps, d, psi, xi1, xi2):
    r"""
    Modified Ergun's equation[^1]:

    $$
    \Delta P = \frac{\Delta z G^2}{\rho_f d} \left[ \xi_1 \frac{(1-\epsilon)^2}{\epsilon^3 \psi^2} \frac{\mu_f}{Gd}
    + \xi_2 frac{(1-\epsilon)}{\epsilon^3 \psi}\right]
    $$

    [^1]: I. Macdonald, M. El-Sayed, K. Mow, and F. Dullien, “Flow through porous media-the Ergun equation revisited,”
    Industrial & Engineering Chemistry Fundamentals, vol. 18, no. 3, pp. 199–208, 1979.

    Parameters:
        dz: Axial position step size [m].
        rho_f: Density of the fluid [kg/m^3].
        mu_f: Dynamic viscosity of the fluid [Pa s].
        G: Effective mass flow rate per unit cross-section (kg/m^2 s).
        eps: Void fraction [-].
        d: Particle diameter [m].
        psi: Sphericity [-].
        xi1: Viscous loss coefficient.
        xi2: Inertial loss coefficient.

    """
    return dz * G**2 / (rho_f * d) * (
            xi1 * (1 - eps)**2 * mu_f / (eps**3 * psi**2 * G * d)  # Viscous loss
            + xi2 * (1 - eps) / (eps**3 * psi)  # Inertial loss
    )