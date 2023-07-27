"""
An implementation of the packed bed thermal energy storage model described by Battisti et al.[^1].

[^1]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy storage
systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal Engineering, vol.
183, p. 116032, 2021.
"""

import numpy as np


def effective_film_thickness_ratio(k_s: float, k_f: float, eps: float):
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

    and $\sin^2 \theta_i = \frac{1}{n_i}$ for $n_1 = 1.5$ and $n_2 = 4\sqrt{3}$.

    [1] D. Kunii and J. Smith, “Heat transfer characteristics of porous rocks,” AIChE Journal, vol. 6, no. 1, pp.
    71–78, 1960.

    Parameters:
        k_s: Thermal conductivity of the solid [W/m K].
        k_f: Thermal conductivity of the fluid [W/m K].
        eps: Void fraction [-].

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


def void_radiative_heat_transfer_coeff(T: float, eps: float, E_s: float):
    r"""
    Calculates the void-to-void radiative heat transfer coefficient for a packed bed given by the correlation of Yagi
    and Kunii[^1]

    $$
    h_{rv} = \frac{0.1952}{1 + \frac{\varepsilon (1-E_s)}{2E_s (1-\varepsilon)}} \left(\frac{T}{100}\right)^3
    $$

    [^1]: S. Yagi and D. Kunii, “Studies on effective thermal conductivities in packed beds,” AIChE Journal,
    vol. 3, no. 3, pp. 373–381, 1957.

    Parameters:
        T: Temperature [K].
        eps: Void fraction [-].
        E_s: Emissivity of the solid [-].

    """
    return 0.1952 * (T/100)**3 / (1 + eps * (1 - E_s) / (2 * E_s * (1 - eps)))


def surface_radiative_heat_transfer_coeff(T: float, E_s: float):
    r"""
    Calculates the surface-to-surface radiative heat transfer coefficient for a packed bed given by the correlation of
    Yagi and Kunii[^1]

    $$
    h_{rs} = \frac{0.1952E_s}{2-E_s} \left( \frac{T}{100}\right)^3
    $$

    [^1]: S. Yagi and D. Kunii, “Studies on effective thermal conductivities in packed beds,” AIChE Journal,
    vol. 3, no. 3, pp. 373–381, 1957.

    Parameters:
        T: Temperature [K].
        E_s: Emissivity of the solid [-].

    """
    return 0.1952 * (T/100)**3 * E_s / (2 - E_s)


def effective_thermal_conductivity(
        k_f: float,
        k_s: float,
        eps: float,
        h_rv: float,
        h_rs: float,
        phi: float,
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

    [^1]: D. Kunii and J. Smith, “Heat transfer characteristics of porous rocks,” AIChE Journal, vol. 6, no. 1,
    pp. 71–78, 1960.

    Parameters:
        k_f: Thermal conductivity of the fluid [W/mK].
        k_s: Thermal conductivity of the solid [W/mK].
        eps: Void fraction [-].
        h_rv: Void-to-void radiative heat transfer coefficient.
        h_rs: Surface-to-surface radiative heat transfer coefficient.
        phi: Effective film thickness ratio.
        d: Solid particle diameter [m].
        beta: Ratio of the average length between the centers of two neighboring solids to
            the mean particle diameter.

    """
    return k_f * (
        eps * (1 + beta * h_rv * d / k_f) +
        beta * (1 - eps) / (
                1 / (1 / phi + h_rs * d / k_f)
                + 2/3 * k_f / k_s)
    )


def volumetric_convective_heat_transfer_coeff(k_f, cp_f, G, eps, d):
    r"""
    Calculates the volumetric convective heat transfer coefficient as the product of the convective heat
    transfer coefficient for a spherical particle and the ratio between the particles total heat transfer area
    and the total volume:

    $$
    h_v = h_part \frac{6(1-\varepsilon)}{d}
    $$

    The particle convective heat transfer coefficient is given by Pfeffer's correlation[^1]:

    $$
    h_{part} = 1.26 \left[\frac{1-(1-\varepsilon)^{5/3}}{W}\right]^{1/3} (c_{p_f} G)^{1/3}
    \left(\frac{k_f}{d}\right)^{2/3}
    $$

    where $W$ is given by:

    $$
    W = 2-3(1-\varepsilon)^{1/3}+3(1-\varepsilon)^{5/3}-2(1-\varepsilon)^2
    $$

    The lower limit of analytical heat transfer is given by:

    $$
    h_{part} = \frac{2k_f}{d}
    $$

    for a heated isothermal sphere in a quiescent fluid medium.

    [^1]: R. Pfeffer, “Heat and mass transport in multiparticle systems,” Industrial & Engineering Chemistry
    Fundamentals, vol. 3, no. 4, pp. 380–383, 1964.

    Parameters:
        k_f: Thermal conductivity of the fluid [W/m K].
        cp_f: Specific heat capacity of the fluid [J/kg K].
        G: Effective mass flow rate per unit of cross-section [kg/m^2 s].
        eps: Void fraction [-].
        d: Particle diameter [m].
    """

    W = 2 - 3 * (1 - eps)**(1/3) + 3 * (1 - eps)**(5/3) - 2 * (1 - eps)**2
    h_part = np.max(
        1.26 * ((1 - (1 - eps)**(5/3)) / W)**(1/3) * (cp_f * G)**(1/3) * (k_f / d)**(2/3),  # Pfeffer's correlation
        2 * k_f / d  # Lower limit
    )
    return h_part * 6 * (1 - eps) / d


def conv_wall_heat_transfer_coeff(k_f, Re_d, Pr, d):
    r"""
    Returns the convective heat transfer coefficient between the fluid and the wall:

    $$
    h_{wall}^{cv} = \left( 2.58 Re_d^{1/3} Pr^{1/3} + 0.094 Re_d^{0.8} Pr^{0.4} \right) \frac{k_f}{d}
    $$

    according to the correlation of Beek[^1].

    [^1]: J. Beek, “Design of packed catalytic reactors,” in Advances in Chemical Engineering, Elsevier,
    1962, pp. 203–271.

    Parameters:
        k_f: Thermal conductivity of the fluid [W/m K].
        Re_d: Reynolds number of the flow based on particle diameter.
        Pr: Prandtl number of the flow.
        d: Particle diameter [W].

    """
    return (2.58 * Re_d**(1/3) * Pr**(1/3) + 0.094 * Re_d**0.8 * Pr**0.4) * k_f / d


def cond_rad_wall_heat_transfer_coeff(k_f, k_s, h_rv, h_rs, eps, d, phi):
    r"""
    Calculates the conductive and radiative wall heat transfer coefficient according to Ofuchi and Kunii[^1]

    $$
    k_{eff}^{stag} = k_f \left[\varepsilon \left(1 + \frac{h_{rv}d}{k_f}\right) +
    \frac{1-\varepsilon}{\left(\frac{1}{\phi}+\frac{h_{rs}d}{k_f}\right)^{-1} + \frac{2}{3\kappa}}\right]
    $$

    and

    $$
    k_{wall}^{stag} = k_f \left[\varepsilon_{wall} \left(2 + \frac{h_{rv}d}{k_f}\right) +
    \frac{1-\varepsilon_{wall}}{\left(\frac{1}{\phi_{wall}}+\frac{h_{rs}d}{k_f}\right)^{-1} + \frac{1}{3\kappa}}\right]
    $$

    where the wall porosity ($\varepsilon_{wall}$) is assumed to be $0.4$ and $\phi_{wall}$ is given by

    $$
    \phi_{wall} = \frac{1}{4} \frac{\left(\frac{\kappa-1}{\kappa}\right)^2}{\ln{\kappa} - \frac{\kappa-1}{\kappa}} -
    \frac{1}{3\kappa}
    $$

    [^1]: K. Ofuchi and D. Kunii, “Heat-transfer characteristics of packed beds with stagnant fluids,” International
    Journal of Heat and Mass Transfer, vol. 8, no. 5, pp. 749–757, 1965.

    Parameters:
        k_f: Thermal conductivity of the fluid [W/m K].
        k_s: Thermal conductivity of the solid [W/m K].
        h_rv: void-to-void radiative heat transfer coefficient.
        h_rs: Solid-to-solid radiative heat transfer coefficient
        eps: Void fraction [-].
        d: Particle diameter [m].
        phi: Effective film thickness ratio.

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
