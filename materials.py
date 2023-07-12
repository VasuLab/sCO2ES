class Alumina:
    """
    Temperature-dependent physical properties of alumina (Al2O3). Input temperatures must be in [K].
    Equations found in Appendix C.1 of the reference:

    F. Battisti, L. de Araujo Passos, and A. da Silva, “Performance mapping of packed-bed thermal energy storage
    systems for concentrating solar-powered plants using supercritical carbon dioxide,” Applied Thermal Engineering,
    vol. 183, p. 116032, 2021.
    """

    @staticmethod
    def emissivity(T):
        """
        Calculates the temperature-dependent emissivity.

        Reference:
        M. E. Whitson Jr, "Handbook of the Infrared Optical Properties of Al2O3. Carbon, MGO and ZrO2. Volume 1,"
        El Segundo/CA, 1975.
        """
        T_star = (T - 953.8151) / 432.1046
        return 0.5201 - 0.1794 * T_star + 0.01343 * T_star ** 2 + 0.01861 * T_star ** 3

    @staticmethod
    def thermal_conductivity(T):
        """
        Calculates the temperature-dependent thermal conductivity [W/m K].

        Reference:
        "AETG/UC San Diego," [Online]. Available: www.ferp.ucsd.edu/LIB/PROPS/PANOS/al2o3.html
        """
        return 85.686 - 0.22972 * T + 2.607e-4 * T ** 2 - 1.3607e-7 * T ** 3 + 2.7092e-11 * T ** 4
