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
