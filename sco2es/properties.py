from typing import Protocol
from abc import abstractmethod


class SolidProperties(Protocol):
    """Protocol class defining the required interface for updating temperature-dependent solid properties
    for the [`PackedBed`][sco2es.PackedBed] model.
    """

    density: float
    r"""Density, $\rho$ [kg/m^3^]."""

    @staticmethod
    @abstractmethod
    def internal_energy(T):
        """Internal energy, $e$ [J/kg]."""

    @staticmethod
    @abstractmethod
    def internal_energy_linear_coeffs(T):
        r"""
        Coefficients, $\alpha_1$ [J/kg⋅K] and $\alpha_2$ [J/kg], for the linearized expression
        of internal energy:

        $$
        e(T) = \alpha_1 T + \alpha_2
        $$
        """

    @staticmethod
    @abstractmethod
    def thermal_conductivity(T):
        """Thermal conductivity, $k$ [W/m⋅K]."""

    @staticmethod
    @abstractmethod
    def emissivity(T):
        r"""Emissivity, $E$ [-]."""


class Alumina(SolidProperties):
    """An implementation of experimental correlations for alumina's relevant properties."""

    density = 3950

    @staticmethod
    def internal_energy(T):
        """
        Internal energy of alumina [^1]^,^[^2].

        [^1]: K. K. Kelley, "Contributions to the data on theoretical metallurgy, XIII. High-temperature heat-content,
        heat-capacity, and entropy data for the elements and inorganic compounds," in Bulletin 584 Bureau of Mines,
        1960.
        [^2]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy
        storage systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal
        Engineering, vol. 183, p. 116032, 2021.
        """
        psi1, psi2, psi3, psi4 = 1.712e3, 0.658, 6.750e-5, -2.010e4
        T_ref = 25 + 273
        return psi1 * (psi2 * (T - T_ref) + psi3 * (T**2 - T_ref**2) / 2 - psi4 * (1 / T - 1 / T_ref))

    @staticmethod
    def internal_energy_linear_coeffs(T):
        """
        Linearized internal energy coefficients of alumina [^1]^,^[^2].

        [^1]: K. K. Kelley, "Contributions to the data on theoretical metallurgy, XIII. High-temperature heat-content,
        heat-capacity, and entropy data for the elements and inorganic compounds," in Bulletin 584 Bureau of Mines,
        1960.
        [^2]: F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy
        storage systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal
        Engineering, vol. 183, p. 116032, 2021.
        """
        psi1, psi2, psi3, psi4 = 1.712e3, 0.658, 6.750e-5, -2.010e4
        T_ref = 25 + 273
        alpha1 = psi1 * (psi2 + psi3 * T + psi4 / T ** 2)
        alpha2 = -psi1 * (psi2 * T_ref + psi3 * (T ** 2 + T_ref ** 2) / 2 + psi4 * (2 / T - 1 / T_ref))
        return alpha1, alpha2

    @staticmethod
    def thermal_conductivity(T):
        """
        Thermal conductivity of alumina[^1].

        [^1]: "AETG/UC San Diego," [Online]. Available: www.ferp.ucsd.edu/LIB/PROPS/PANOS/al2o3.html
        """
        return 85.686 - 0.22972 * T + 2.607e-4 * T**2 - 1.3607e-7 * T**3 + 2.7092e-11 * T**4

    @staticmethod
    def emissivity(T):
        """
        Emissivity[^1] of alumina.

        [^1]: M. E. Whitson Jr, "Handbook of the Infrared Optical Properties of Al2O3. Carbon, MGO and ZrO2. Volume 1,"
        El Segundo/CA, 1975.
        """
        T = (T - 953.8151) / 432.1046
        return 0.5201 - 0.1794 * T + 0.01343 * T**2 + 0.01861 * T**3
